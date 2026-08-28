from typing import List

from pydantic import BaseModel, Field


class WorkloadProfile(BaseModel):
    num_records: int = Field(..., gt=0, examples=[1_000_000])
    skewness: float = Field(..., ge=0.0, le=1.0, examples=[0.3])
    joins: int = Field(..., ge=0, le=5, examples=[1])
    group_cardinality: int = Field(..., gt=0, examples=[1000])
    num_aggregations: int = Field(..., ge=1, le=4, examples=[2])
    has_orderby: int = Field(..., ge=0, le=1, examples=[0])
    filter_selectivity: float = Field(..., ge=0.0, le=1.0, examples=[0.7])
    top_n: int = Field(5, ge=1, le=20)


class ConfigRecommendation(BaseModel):
    cores: int
    executor_memory_gb: int
    partitions: int
    cache_enabled: int
    predicted_time_sec: float
    partition_efficiency: float
    data_size_gb: float
    shuffle_size_mb: float
    shuffle_intensity: float
    memory_pressure: float


class RecommendationResponse(BaseModel):
    workload: WorkloadProfile
    recommendations: List[ConfigRecommendation]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    features: List[str]
    n_candidate_configs: int
