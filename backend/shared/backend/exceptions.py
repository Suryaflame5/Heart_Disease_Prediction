from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

class APIError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.message}
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        errors.append(f"[{loc}]: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": "; ".join(errors)}
    )

async def generic_exception_handler(request: Request, exc: Exception):
    error_msg = f"Internal Server Error: {str(exc)}"
    # Log full traceback locally (we assume logging configuration prints stack traces)
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": error_msg}
    )
