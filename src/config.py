import tomllib
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field, model_validator

CONFIG_PATH = Path(__file__).parent / "config.toml"


class Data(BaseModel):
    expected_wire_width: float = Field(..., gt=0)
    smallest_wire_ratio: float = Field(..., ge=0)
    largest_wire_ratio: float = Field(..., ge=0)
    wires_per_std_dev: float = Field(..., ge=1)
    samples_across: int = Field(..., ge=1)
    z_score_filter_threshold: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate(self):
        if self.smallest_wire_ratio >= self.largest_wire_ratio:
            raise ValueError("smallest_wire_ratio must be less than largest_wire_ratio")
        return self


class App(BaseModel):
    window_size: int = Field(..., ge=400)
    next_key: str


class ScaleBarDetection(BaseModel):
    width_offset: float
    min_area_ratio: float = Field(..., ge=0)
    max_area_ratio: float = Field(..., le=1)
    min_brightness_threshold: int = Field(..., ge=0, le=255)
    max_brightness_threshold: int = Field(..., ge=0, le=255)

    @model_validator(mode="after")
    def validate(self):
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be less than max_area_ratio")
        if self.min_brightness_threshold >= self.max_brightness_threshold:
            raise ValueError(
                "min_brightness_threshold must be less than max_brightness_threshold"
            )
        return self


class MaskPainter(BaseModel):
    default_radius: int = Field(..., ge=2)
    paint_color: list[Annotated[int, Field(ge=0, le=255)]] = Field(
        ..., min_length=3, max_length=3
    )
    paint_opacity: float = Field(..., ge=0, le=1)
    scroll_factor: float = Field(..., gt=0)


class Settings(BaseModel):
    data: Data
    app: App
    scale_bar_detection: ScaleBarDetection
    mask_painter: MaskPainter


with open(CONFIG_PATH, "rb") as f:
    data = tomllib.load(f)
    settings = Settings(**data)
