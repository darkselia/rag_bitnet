import uvicorn
from common import config


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
