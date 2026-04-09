#!/usr/bin/env python3
"""
GitHub API client for repository walking and file content extraction.
"""

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator, Tuple
from urllib.parse import urlparse

import requests


@dataclass
class GitHubRepoInfo:
    """Parsed GitHub repository information."""

    owner: str
    repo: str
    branch: str = "main"
    path: str = ""

    def __str__(self) -> str:
        return f"{self.owner}/{self.repo}/{self.branch}/{self.path}"


@dataclass
class GitHubFile:
    """GitHub file information."""

    path: str
    name: str
    size: int
    download_url: Optional[str] = None
    content: Optional[str] = None


class GitHubClient:
    """GitHub API client for repository operations."""

    BASE_API_URL = "https://api.github.com"
    BASE_RAW_URL = "https://raw.githubusercontent.com"

    def __init__(self, token: Optional[str] = None, timeout: int = 30):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (optional, for higher rate limits)
            timeout: Request timeout in seconds
        """
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()

        # Set up headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Prompt-Detective/1.0",
        }

        # Try to get token from environment if not provided
        if not token:
            token = os.environ.get("GITHUB_TOKEN")

        if token:
            self.token = token
            self.headers["Authorization"] = f"token {token}"
            print(f"🔑 Using GitHub token (rate limit: 5,000 requests/hour)")
        else:
            print(f"⚠️  No GitHub token provided (rate limit: 60 requests/hour)")

    def parse_github_url(self, url: str) -> GitHubRepoInfo:
        """
        Parse GitHub URL to extract repository information.

        Supports:
        - https://github.com/{owner}/{repo}
        - https://github.com/{owner}/{repo}/tree/{branch}
        - https://github.com/{owner}/{repo}/tree/{branch}/{path}
        - https://github.com/{owner}/{repo}/blob/{branch}/{path}
        - https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}

        Args:
            url: GitHub URL

        Returns:
            GitHubRepoInfo object

        Raises:
            ValueError: If URL is not a valid GitHub URL
        """
        # Parse URL
        parsed = urlparse(url)

        # Check if it's a GitHub URL
        if parsed.netloc not in [
            "github.com",
            "api.github.com",
            "raw.githubusercontent.com",
        ]:
            raise ValueError(f"Not a GitHub URL: {url}")

        # Split path components
        path_parts = parsed.path.strip("/").split("/")

        if parsed.netloc == "raw.githubusercontent.com":
            # raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
            if len(path_parts) < 3:
                raise ValueError(f"Invalid raw GitHub URL: {url}")

            owner = path_parts[0]
            repo = path_parts[1]
            branch = path_parts[2]
            path = "/".join(path_parts[3:]) if len(path_parts) > 3 else ""

        elif parsed.netloc == "api.github.com":
            # api.github.com/repos/{owner}/{repo}/contents/{path}
            if len(path_parts) < 3 or path_parts[0] != "repos":
                raise ValueError(f"Invalid GitHub API URL: {url}")

            owner = path_parts[1]
            repo = path_parts[2]

            # Extract branch and path from query parameters or path
            branch = "main"
            path = ""

            if "contents" in path_parts:
                content_idx = path_parts.index("contents")
                if len(path_parts) > content_idx + 1:
                    path = "/".join(path_parts[content_idx + 1 :])

            # Check for ref in query parameters
            if parsed.query:
                query_params = dict(
                    qp.split("=") for qp in parsed.query.split("&") if "=" in qp
                )
                if "ref" in query_params:
                    branch = query_params["ref"]

        else:  # github.com
            # github.com/{owner}/{repo}
            if len(path_parts) < 2:
                raise ValueError(f"Invalid GitHub URL: {url}")

            owner = path_parts[0]
            repo = path_parts[1]
            branch = "main"
            path = ""

            # Check for tree/blob paths
            if len(path_parts) >= 4 and path_parts[2] in ["tree", "blob"]:
                branch = path_parts[3]
                if len(path_parts) > 4:
                    path = "/".join(path_parts[4:])

        return GitHubRepoInfo(owner=owner, repo=repo, branch=branch, path=path)

    def _make_request(
        self, url: str, method: str = "GET", **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with rate limiting and error handling.

        Args:
            url: Request URL
            method: HTTP method
            **kwargs: Additional request arguments

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure
        """
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self.headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()

            # Check rate limits
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                limit = int(response.headers.get("X-RateLimit-Limit", 60))

                if remaining < 10:
                    print(
                        f"⚠️  Warning: GitHub API rate limit low ({remaining}/{limit} requests remaining)"
                    )

                # Show authentication status
                if remaining == limit - 1:  # First request
                    if self.token:
                        print(
                            f"🔑 Authenticated with GitHub token (rate limit: {limit}/hour)"
                        )
                    else:
                        print(
                            f"⚠️  Unauthenticated (rate limit: {limit}/hour) - consider using --github-token or GITHUB_TOKEN env var"
                        )

            return response

        except requests.exceptions.RequestException as e:
            if hasattr(e.response, "status_code"):
                if e.response.status_code == 403:
                    if "rate limit" in e.response.text.lower():
                        reset_time = e.response.headers.get("X-RateLimit-Reset")
                        if reset_time:
                            reset_time = int(reset_time)
                            wait_time = max(0, reset_time - time.time())
                            print(
                                f"⏳ Rate limited. Waiting {wait_time:.0f} seconds..."
                            )
                            time.sleep(wait_time + 1)
                            return self._make_request(url, method, **kwargs)
                    else:
                        # Other 403 errors
                        print(f"❌ Access denied (403): {e.response.text[:200]}")
                        if not self.token:
                            print(
                                "💡 Tip: Try using --github-token or set GITHUB_TOKEN environment variable"
                            )
                        raise

                elif e.response.status_code == 404:
                    raise FileNotFoundError(f"Resource not found: {url}")

                elif e.response.status_code == 401:
                    print(f"❌ Authentication failed (401)")
                    if self.token:
                        print(
                            "💡 Tip: Check if your GitHub token is valid and has appropriate permissions"
                        )
                    raise

            raise

    def get_contents(
        self, owner: str, repo: str, path: str = "", ref: str = "main"
    ) -> List[Dict]:
        """
        Get contents of a directory or file from GitHub.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path within repository (empty for root)
            ref: Git reference (branch, tag, or commit)

        Returns:
            List of content items (files and directories)

        Raises:
            FileNotFoundError: If path doesn't exist
            requests.RequestException: On API error
        """
        url = f"{self.BASE_API_URL}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}

        response = self._make_request(url, params=params)
        data = response.json()

        # If it's a single file, return it as a list with one item
        if isinstance(data, dict):
            return [data]

        return data

    def get_file_content(
        self, owner: str, repo: str, path: str, ref: str = "main"
    ) -> str:
        """
        Get content of a file from GitHub.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path within repository
            ref: Git reference (branch, tag, or commit)

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If content is not text or encoding fails
        """
        # Try to get raw content first (faster and doesn't require base64 decoding)
        raw_url = f"{self.BASE_RAW_URL}/{owner}/{repo}/{ref}/{path}"
        try:
            response = self._make_request(raw_url)
            return response.text
        except (requests.RequestException, FileNotFoundError):
            # Fall back to API endpoint
            pass

        # Get content via API
        contents = self.get_contents(owner, repo, path, ref)
        if not contents or len(contents) != 1:
            raise FileNotFoundError(f"File not found: {path}")

        file_info = contents[0]

        # Check if it's actually a file
        if file_info.get("type") != "file":
            raise ValueError(f"Path is not a file: {path}")

        # Decode base64 content
        content = file_info.get("content")
        encoding = file_info.get("encoding")

        if encoding == "base64" and content:
            try:
                decoded = base64.b64decode(content).decode("utf-8")
                return decoded
            except (UnicodeDecodeError, base64.binascii.Error) as e:
                raise ValueError(f"Failed to decode file content: {e}")
        elif content:
            # Content might already be decoded (for small files)
            return content
        else:
            # Try download_url
            download_url = file_info.get("download_url")
            if download_url:
                response = self._make_request(download_url)
                return response.text

        raise ValueError(f"Could not extract content from file: {path}")

    def walk_repository(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "main",
        max_depth: int = 10,
        current_depth: int = 0,
    ) -> Generator[GitHubFile, None, None]:
        """
        Recursively walk through repository directory structure.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Starting path (empty for root)
            ref: Git reference (branch, tag, or commit)
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth (internal use)

        Yields:
            GitHubFile objects for each file found
        """
        if current_depth >= max_depth:
            print(f"⚠️  Max depth reached at: {path}")
            return

        try:
            contents = self.get_contents(owner, repo, path, ref)
        except FileNotFoundError:
            print(f"⚠️  Path not found: {path}")
            return
        except requests.RequestException as e:
            print(f"❌ Error accessing {path}: {e}")
            return

        for item in contents:
            item_type = item.get("type")
            item_path = item.get("path", "")
            item_name = item.get("name", "")
            item_size = item.get("size", 0)
            download_url = item.get("download_url")

            if item_type == "file":
                yield GitHubFile(
                    path=item_path,
                    name=item_name,
                    size=item_size,
                    download_url=download_url,
                )
            elif item_type == "dir":
                # Recursively process directory
                yield from self.walk_repository(
                    owner=owner,
                    repo=repo,
                    path=item_path,
                    ref=ref,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
            # Skip symlinks and submodules

    def get_repository_info(self, owner: str, repo: str) -> Dict:
        """
        Get repository information.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Repository information dictionary
        """
        url = f"{self.BASE_API_URL}/repos/{owner}/{repo}"
        response = self._make_request(url)
        return response.json()


def test_github_client():
    """Test function for GitHub client."""
    client = GitHubClient()

    # Test URL parsing
    test_urls = [
        "https://github.com/openai/openai-python",
        "https://github.com/openai/openai-python/tree/main",
        "https://github.com/openai/openai-python/blob/main/README.md",
        "https://raw.githubusercontent.com/openai/openai-python/main/README.md",
        "https://api.github.com/repos/openai/openai-python/contents/README.md",
    ]

    for url in test_urls:
        try:
            info = client.parse_github_url(url)
            print(f"✓ Parsed: {url}")
            print(f"  -> {info}")
        except ValueError as e:
            print(f"✗ Failed: {url} - {e}")

    # Test getting a file
    try:
        content = client.get_file_content("openai", "openai-python", "README.md")
        print(f"\n✓ Got README.md content ({len(content)} chars)")
        print(f"  Preview: {content[:200]}...")
    except Exception as e:
        print(f"\n✗ Failed to get file: {e}")


if __name__ == "__main__":
    test_github_client()
