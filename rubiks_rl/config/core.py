from pydantic import BaseModel

class Config(BaseModel):
    """Master config object."""


def create_and_validate_config() -> Config:

    
    pass

config = create_and_validate_config()