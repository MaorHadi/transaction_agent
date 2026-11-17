from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Named as on the Alpaca dashboard: "Key" is the key ID, "Client" is the secret.
    alpaca_api_key: str | None = Field(default=None, validation_alias="API_ALPACA_KEY")
    alpaca_secret_key: str | None = Field(default=None, validation_alias="API_ALPACA_CLIENT")
    # Paper or live trading URL. Not needed for market data.
    alpaca_endpoint: str | None = Field(default=None, validation_alias="API_ALPACA_ENDPOINT")
    data_dir: Path = Path("data")
