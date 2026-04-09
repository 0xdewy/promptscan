"""
Security Headers Middleware for Prompt Detective API.

This middleware adds security headers to all responses to enhance security.
"""

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import os


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.environment = os.getenv("ENVIRONMENT", "production")

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(request, response)

        return response

    def _add_security_headers(self, request: Request, response: Response):
        """Add security headers to the response."""

        # Content Security Policy
        # Allow resources from self and trusted CDNs
        csp_policy = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
        ]

        if self.environment == "development":
            # Allow more in development for debugging
            csp_policy = [
                "default-src 'self' 'unsafe-inline'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "font-src 'self' data:",
                "img-src 'self' data:",
                "connect-src 'self'",
            ]

        response.headers["Content-Security-Policy"] = "; ".join(csp_policy)

        # X-Frame-Options: Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection: Enable XSS filtering
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy: Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy: Control browser features
        permissions_policy = [
            "camera=()",
            "microphone=()",
            "geolocation=()",
            "payment=()",
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions_policy)

        # Strict-Transport-Security (HSTS) - only in production with HTTPS
        if self.environment == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Cache-Control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        else:
            # Static files can be cached
            response.headers["Cache-Control"] = "public, max-age=3600"

        # X-Powered-By: Remove server information
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        # Server: Remove server information
        if "Server" in response.headers:
            del response.headers["Server"]


def setup_security_headers(app: FastAPI) -> FastAPI:
    """Setup security headers middleware for the FastAPI app."""
    app.add_middleware(SecurityHeadersMiddleware)
    return app
