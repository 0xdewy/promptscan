"""
x402 Payment Middleware for Prompt Detective API.

This middleware verifies x402 payments for API endpoints that require payment.
It checks for valid payment headers and verifies them with the x402 facilitator.

Endpoints that should be payment-gated:
- POST /api/v1/predict - Main prediction endpoint
- POST /api/v1/predict/batch - Batch prediction endpoint

Endpoints that should be free:
- GET / - Frontend
- GET /static/* - Static files
- GET /api/v1/health - Health check
- GET /api/v1/info - API info
- GET /api/v1/stats - Statistics
- GET /api/v1/feedback/stats - Feedback statistics
- POST /api/v1/feedback - Feedback submission (free for users to provide feedback)
- POST /api/v1/upload - File upload (frontend-only, protected by API key)
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class X402PaymentMiddleware(BaseHTTPMiddleware):
    """Middleware to verify x402 payments for API endpoints."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.free_endpoints = [
            "/",
            "/api/v1/health",
            "/api/v1/info",
            "/api/v1/stats",
            "/api/v1/feedback",
            "/api/v1/feedback/stats",
            "/static/",
        ]
        self.payment_required_endpoints = [
            "/api/v1/predict",
            "/api/v1/predict/batch",
        ]

        # Load x402 configuration from environment
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        self.facilitator_url = os.getenv(
            "FACILITATOR_URL", "https://api.cdp.coinbase.com/platform/v2/x402"
        )
        self.network = os.getenv("NETWORK", "eip155:8453")
        self.price = os.getenv("PRICE", "$0.01")

        # Check if x402 is enabled
        self.x402_enabled = os.getenv("X402_ENABLED", "true").lower() == "true"

        logger.info(f"X402 Payment Middleware initialized")
        logger.info(f"Wallet: {self.wallet_address}")
        logger.info(f"Facilitator: {self.facilitator_url}")
        logger.info(f"Network: {self.network}")
        logger.info(f"Price: {self.price}")
        logger.info(f"Enabled: {self.x402_enabled}")

        if not self.wallet_address or self.wallet_address == "0xYourWalletAddressHere":
            logger.warning("WALLET_ADDRESS not properly configured in .env file")

    async def dispatch(self, request: Request, call_next):
        """Process incoming request and verify payment if required."""

        # Check if this is a free endpoint
        if self._is_free_endpoint(request.url.path):
            return await call_next(request)

        # Check if this endpoint requires payment
        if not self._requires_payment(request.url.path):
            return await call_next(request)

        # If x402 is disabled, allow requests without payment
        if not self.x402_enabled:
            logger.debug(f"X402 disabled, allowing request to {request.url.path}")
            return await call_next(request)

        # Check if this is a frontend request (same origin)
        # Frontend should be able to use the API for free
        if self._is_frontend_request(request):
            logger.debug(
                f"Frontend request detected, allowing without payment: {request.url.path}"
            )
            return await call_next(request)

        # Check for payment headers
        payment_header = request.headers.get("PAYMENT")
        payment_required_header = request.headers.get("PAYMENT-REQUIRED")

        # If no payment header, return 402 Payment Required
        if not payment_header:
            return self._create_payment_required_response()

        # Verify the payment
        try:
            is_valid = await self._verify_payment(payment_header, request)
            if is_valid:
                logger.info(f"Valid payment verified for {request.url.path}")
                return await call_next(request)
            else:
                logger.warning(f"Invalid payment for {request.url.path}")
                return self._create_payment_required_response()
        except Exception as e:
            logger.error(f"Payment verification error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Payment verification failed", "detail": str(e)},
            )

    def _is_free_endpoint(self, path: str) -> bool:
        """Check if the endpoint is in the free list."""
        # Exact match
        if path in self.free_endpoints:
            return True

        # Prefix match for static files
        if path.startswith("/static/"):
            return True

        # Check for frontend routes
        if path == "/" or path.startswith("/?"):
            return True

        return False

    def _requires_payment(self, path: str) -> bool:
        """Check if the endpoint requires payment."""
        for endpoint in self.payment_required_endpoints:
            if path == endpoint:
                return True
        return False

    def _is_frontend_request(self, request: Request) -> bool:
        """Check if request is from the frontend (same origin)."""
        # Get the origin header
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")

        # Get server host and port
        server_host = request.url.hostname
        server_port = request.url.port or 8000

        # For development/testing, be more permissive
        # In production, this should be stricter

        # Check if this looks like a browser request
        user_agent = request.headers.get("user-agent", "").lower()
        is_browser = any(
            browser in user_agent
            for browser in ["mozilla", "chrome", "safari", "firefox", "edge"]
        )

        # Check origin header
        if origin:
            try:
                from urllib.parse import urlparse

                origin_parsed = urlparse(origin)
                origin_host = origin_parsed.hostname

                # For development, allow localhost origins
                if origin_host in ["localhost", "127.0.0.1"]:
                    logger.debug(f"Localhost origin detected: {origin}")
                    return True

                # Compare with server host (exact match)
                if origin_host == server_host:
                    logger.debug(f"Same-origin request from {origin}")
                    return True

                # For development, also allow testserver (TestClient)
                if origin_host == "testserver" and server_host == "testserver":
                    logger.debug(f"Testserver origin detected: {origin}")
                    return True

            except Exception as e:
                logger.debug(f"Error parsing origin {origin}: {e}")

        # Check referer header
        if referer:
            try:
                from urllib.parse import urlparse

                referer_parsed = urlparse(referer)
                referer_host = referer_parsed.hostname

                # For development, allow localhost referers
                if referer_host in ["localhost", "127.0.0.1"]:
                    logger.debug(f"Localhost referer detected: {referer}")
                    return True

                # Compare with server host
                if referer_host == server_host:
                    logger.debug(f"Same-origin request from referer {referer}")
                    return True

                # For development, also allow testserver (TestClient)
                if referer_host == "testserver" and server_host == "testserver":
                    logger.debug(f"Testserver referer detected: {referer}")
                    return True

            except Exception as e:
                logger.debug(f"Error parsing referer {referer}: {e}")

        # For development/localhost/testserver, allow all browser requests
        if (
            server_host in ["localhost", "127.0.0.1", "0.0.0.0", "testserver"]
            and is_browser
        ):
            logger.debug(f"Browser request on {server_host}: {user_agent[:50]}...")
            return True

        # Check if request has frontend-specific headers
        # Frontend JavaScript might send specific headers
        if is_browser:
            logger.debug(f"Browser user-agent detected: {user_agent[:50]}...")
            # This is likely a frontend request
            return True

        # Check for frontend-specific content type
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type and is_browser:
            logger.debug(f"Browser JSON request: {content_type}")
            return True

        logger.debug(
            f"Not a frontend request: origin={origin}, referer={referer}, host={server_host}, UA={user_agent[:50]}"
        )
        return False

    def _create_payment_required_response(self) -> JSONResponse:
        """Create a 402 Payment Required response with payment details."""
        payment_data = {
            "wallet_address": self.wallet_address,
            "facilitator_url": self.facilitator_url,
            "network": self.network,
            "price": self.price,
            "description": "Prompt injection detection service",
            "instructions": "Use x402 client to make payment: x402 pay <url> --prompt 'your prompt'",
        }

        # Encode payment data as base64 for PAYMENT-REQUIRED header
        payment_data_json = json.dumps(payment_data)
        payment_header = base64.b64encode(payment_data_json.encode()).decode()

        # Build headers dictionary, only adding non-None values
        headers = {
            "PAYMENT-REQUIRED": payment_header,
        }

        # Add optional headers if they have values
        if self.wallet_address:
            headers["X-Payment-Wallet"] = self.wallet_address
        if self.price:
            headers["X-Payment-Price"] = self.price
        if self.network:
            headers["X-Payment-Network"] = self.network

        return JSONResponse(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            content={
                "error": "Payment required",
                "message": "This endpoint requires payment via x402 protocol",
                "payment_details": payment_data,
            },
            headers=headers,
        )

    async def _verify_payment(self, payment_header: str, request: Request) -> bool:
        """Verify x402 payment using the x402 SDK."""
        try:
            # Try to import x402 SDK
            from x402 import verify_payment

            # Decode payment header
            try:
                payment_data = json.loads(base64.b64decode(payment_header).decode())
            except:
                logger.error("Failed to decode payment header")
                return False

            # Extract payment information
            payment_id = payment_data.get("payment_id")
            transaction_hash = payment_data.get("transaction_hash")
            amount = payment_data.get("amount")

            if not payment_id and not transaction_hash:
                logger.error("No payment_id or transaction_hash in payment header")
                return False

            # Verify payment with x402 facilitator
            verification_result = await verify_payment(
                payment_id=payment_id,
                transaction_hash=transaction_hash,
                amount=amount,
                facilitator_url=self.facilitator_url,
                network=self.network,
                expected_recipient=self.wallet_address,
            )

            if verification_result.get("verified"):
                logger.info(f"Payment verified: {verification_result}")
                return True
            else:
                logger.warning(f"Payment not verified: {verification_result}")
                return False

        except ImportError:
            logger.warning("x402 SDK not available, using mock verification")
            # For development/testing, allow mock payments
            return self._mock_verify_payment(payment_header)
        except Exception as e:
            logger.error(f"Payment verification exception: {e}")
            return False

    def _mock_verify_payment(self, payment_header: str) -> bool:
        """Mock payment verification for development/testing."""
        try:
            # Try to decode the payment header
            payment_data = json.loads(base64.b64decode(payment_header).decode())

            # Check for mock payment indicator
            if payment_data.get("mock_payment") == "true":
                logger.info("Mock payment accepted for development")
                return True

            # Check for test payment
            if payment_data.get("test") == "true":
                logger.info("Test payment accepted")
                return True

            # In mock mode, accept any valid base64 JSON
            logger.info("Mock verification passed (development mode)")
            return True
        except:
            logger.warning("Mock verification failed - invalid payment header")
            return False


def setup_x402_middleware(app: FastAPI) -> FastAPI:
    """Setup x402 payment middleware for the FastAPI app."""
    # Add x402 middleware
    app.add_middleware(X402PaymentMiddleware)

    # Add endpoint to get payment information
    @app.get("/api/v1/payment/info")
    async def get_payment_info():
        """Get payment information for the API."""
        wallet_address = os.getenv("WALLET_ADDRESS")
        facilitator_url = os.getenv(
            "FACILITATOR_URL", "https://api.cdp.coinbase.com/platform/v2/x402"
        )
        network = os.getenv("NETWORK", "eip155:8453")
        price = os.getenv("PRICE", "$0.01")

        return {
            "payment_required": True,
            "wallet_address": wallet_address,
            "facilitator_url": facilitator_url,
            "network": network,
            "price": price,
            "endpoints_requiring_payment": [
                "/api/v1/predict",
                "/api/v1/predict/batch",
            ],
            "free_endpoints": [
                "/",
                "/api/v1/health",
                "/api/v1/info",
                "/api/v1/stats",
                "/api/v1/feedback",
                "/api/v1/feedback/stats",
                "/static/",
            ],
            "instructions": "Use x402 client: x402 pay http://your-domain/api/v1/predict --prompt 'your prompt'",
        }

    return app
