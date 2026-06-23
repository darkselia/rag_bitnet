from typing import List, Optional

from pydantic import BaseModel

from schemas.app_schema import Message, Timings


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class GenerationSettings(BaseModel):
    n_ctx: int
    n_predict: int
    model: str
    seed: int
    seed_cur: int
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_nl: bool
    stop: List[str]
    max_tokens: int
    n_keep: int
    n_discard: int
    ignore_eos: bool
    stream: bool
    n_probs: int
    min_keep: int
    grammar: str
    samplers: List[str]


class CompletionStreamChunk(BaseModel):
    content: str
    stop: bool
    id_slot: Optional[int] = None
    multimodal: Optional[bool] = None
    index: Optional[int] = None


class CompletionResponse(BaseModel):
    content: str
    id_slot: int = 0
    stop: bool
    model: str = ""
    tokens_predicted: int = 0
    tokens_evaluated: int = 0
    generation_settings: Optional[GenerationSettings] = None
    prompt: str = ""
    has_new_line: bool = False
    truncated: bool = False
    stopped_eos: bool = False
    stopped_word: bool = False
    stopped_limit: bool = False
    stopping_word: str = ""
    tokens_cached: int = 0
    timings: Optional[Timings] = None
    index: int = 0


class ChatCompletionDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: Optional[int] = None
    model: Optional[str] = None
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ChatCompletionChoiceNonStreaming(BaseModel):
    finish_reason: str
    index: int
    message: Message


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoiceNonStreaming]
    usage: Optional[Usage] = None
