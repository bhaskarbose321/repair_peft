from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import re

app = FastAPI()


class Policy(BaseModel):
    minQuality: float
    freshnessRequired: bool
    maxLatencyMs: int
    maxMemoryMb: int
    maxLabeledExamples: int
    maxTotalCost: float
    horizonRequests: int


class Candidate(BaseModel):
    name: str
    available: bool
    quality: float
    freshness: bool
    latencyMs: int
    memoryMb: int
    labeledExamples: int
    oneTimeCost: float
    recurringCost: float


class Token(BaseModel):
    id: int
    role: str
    padding: bool
    text: str


class Parameter(BaseModel):
    name: str
    target: str
    numel: int


class Checkpoint(BaseModel):
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    step: Any = None
    rng: Any = None
    dataPosition: Any = None


class ChooseRequest(BaseModel):
    operation: str
    policy: Policy
    candidates: List[Candidate]


class RepairRequest(BaseModel):
    operation: str
    tokens: List[Token]
    templateApplications: int
    parameters: List[Parameter]
    allowedTargets: List[str]
    inferenceMode: bool
    trainRowIds: List[str]
    evalRowIds: List[str]
    dropoutActiveDuringEval: bool
    artifactFiles: List[str]
    baseRevision: str
    datasetDigest: str
    codeDigest: str
    configDigest: str
    expectedDigests: Dict[str, str]
    microBatch: int
    gradientAccumulation: int
    replicas: int
    expectedEffectiveBatch: int
    checkpoint: Checkpoint
    uninterruptedWeights: List[float]
    resumedWeights: List[float]
    resumeTolerance: float


def is_safe_integer(n: int) -> bool:
    """Check if integer is within safe range (2^53 - 1)"""
    return -9007199254740991 <= n <= 9007199254740991


def is_safe_float(n: float) -> bool:
    """Check if float is finite"""
    import math
    return math.isfinite(n)


def round_to_12_decimals(value: float) -> float:
    """Round to exactly 12 decimal places"""
    return round(value, 12)


def utf8_byte_sort(strings: List[str]) -> List[str]:
    """Sort strings by UTF-8 byte order"""
    return sorted(strings, key=lambda x: x.encode('utf-8'))


def calculate_total_cost(one_time: float, horizon: int, recurring: float) -> float:
    """Calculate total cost rounded to 12 decimal places"""
    total = one_time + horizon * recurring
    return round_to_12_decimals(total)


def validate_candidate(candidate: Candidate, policy: Policy) -> List[str]:
    """Validate a candidate and return reason codes"""
    reasons = []
    
    if not candidate.available:
        reasons.append("UNAVAILABLE")
    
    if candidate.quality < policy.minQuality:
        reasons.append("QUALITY_FLOOR")
    
    if policy.freshnessRequired and not candidate.freshness:
        reasons.append("FRESHNESS_REQUIRED")
    
    if candidate.latencyMs > policy.maxLatencyMs:
        reasons.append("LATENCY_LIMIT")
    
    if candidate.memoryMb > policy.maxMemoryMb:
        reasons.append("MEMORY_LIMIT")
    
    if candidate.labeledExamples > policy.maxLabeledExamples:
        reasons.append("DATA_LIMIT")
    
    total_cost = calculate_total_cost(candidate.oneTimeCost, policy.horizonRequests, candidate.recurringCost)
    if total_cost > policy.maxTotalCost:
        reasons.append("COST_LIMIT")
    
    return sorted(list(set(reasons)), key=lambda x: x.encode('utf-8'))


def handle_choose(request: ChooseRequest) -> Dict[str, Any]:
    """Handle choose operation"""
    priority_order = ["prompt_only", "retrieval", "lora", "qlora"]
    
    total_costs = {}
    reason_codes = {}
    eligible = []
    
    for name in priority_order:
        candidate = None
        for c in request.candidates:
            if c.name == name:
                candidate = c
                break
        
        if candidate is None:
            reason_codes[name] = ["INVALID_INPUT"]
            total_costs[name] = 0.0
            continue
        
        total_costs[name] = calculate_total_cost(
            candidate.oneTimeCost,
            request.policy.horizonRequests,
            candidate.recurringCost
        )
        
        reasons = validate_candidate(candidate, request.policy)
        reason_codes[name] = reasons
        
        if not reasons:
            eligible.append(name)
    
    # Keep eligible in published priority order
    selected = eligible[0] if eligible else None
    
    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes
    }


def validate_token(token: Token) -> bool:
    """Validate a single token"""
    if not is_safe_integer(token.id) or token.id < 0:
        return False
    if token.role not in ["system", "user", "assistant"]:
        return False
    if not isinstance(token.padding, bool):
        return False
    if not isinstance(token.text, str):
        return False
    return True


def compute_labels(tokens: List[Token]) -> List[int]:
    """Compute labels for tokens"""
    labels = []
    all_valid = True
    
    for token in tokens:
        if not validate_token(token):
            all_valid = False
            break
    
    if not all_valid:
        return [-100] * len(tokens)
    
    for token in tokens:
        if token.role == "assistant" and not token.padding:
            labels.append(token.id)
        else:
            labels.append(-100)
    
    return labels


def validate_parameter(param: Parameter) -> bool:
    """Validate a single parameter"""
    if not isinstance(param.name, str):
        return False
    if not isinstance(param.target, str):
        return False
    if not is_safe_integer(param.numel) or param.numel <= 0:
        return False
    return True


def handle_repair(request: RepairRequest) -> Dict[str, Any]:
    """Handle repair operation"""
    reason_codes = []
    labels = []
    template_pass = True
    trainable_params = []
    trainable_count = 0
    peft_config_pass = True
    adapter_files = []
    checkpoint_complete = True
    lineage_pass = True
    eval_isolated = True
    evaluation_deterministic = True
    resume_pass = True
    
    # Token validation
    if not request.tokens:
        reason_codes.append("INVALID_TOKEN")
        labels = [-100] * len(request.tokens)
    else:
        labels = compute_labels(request.tokens)
        if labels == [-100] * len(request.tokens):
            # Check if all tokens are valid but still got -100 labels
            all_valid = all(validate_token(t) for t in request.tokens)
            if not all_valid:
                reason_codes.append("INVALID_TOKEN")
    
    # Template applications
    if request.templateApplications != 1:
        reason_codes.append("CHAT_TEMPLATE_COUNT")
        template_pass = False
    
    # Parameter validation
    param_names = [p.name for p in request.parameters]
    if len(param_names) != len(set(param_names)):
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    else:
        for param in request.parameters:
            if not validate_parameter(param):
                reason_codes.append("INVALID_PARAMETER")
                peft_config_pass = False
                break
    
    # Inference mode
    if request.inferenceMode:
        reason_codes.append("INFERENCE_MODE")
        evaluation_deterministic = False
    
    # Allowed targets
    if not request.allowedTargets:
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    elif len(request.allowedTargets) != len(set(request.allowedTargets)):
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    
    # Check for LoRA parameters
    has_lora = False
    for param in request.parameters:
        if param.target in request.allowedTargets:
            if param.name.endswith(".lora_A.weight") or param.name.endswith(".lora_B.weight"):
                has_lora = True
                break
    
    if not has_lora and request.parameters:
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    
    # Trainable parameters
    if peft_config_pass:
        for param in request.parameters:
            if param.target in request.allowedTargets:
                if param.name.endswith(".lora_A.weight") or param.name.endswith(".lora_B.weight"):
                    trainable_params.append(param.name)
        
        trainable_params = utf8_byte_sort(trainable_params)
        trainable_count = sum(
            p.numel for p in request.parameters 
            if p.name in trainable_params
        )
    
    # Train/eval row IDs
    if not request.trainRowIds or len(request.trainRowIds) != len(set(request.trainRowIds)):
        reason_codes.append("INVALID_PARAMETER")
        eval_isolated = False
    
    if not request.evalRowIds or len(request.evalRowIds) != len(set(request.evalRowIds)):
        reason_codes.append("INVALID_PARAMETER")
        eval_isolated = False
    
    if set(request.trainRowIds) & set(request.evalRowIds):
        reason_codes.append("EVAL_LEAKAGE")
        eval_isolated = False
    
    # Dropout during eval
    if request.dropoutActiveDuringEval:
        reason_codes.append("EVAL_DROPOUT_ACTIVE")
        evaluation_deterministic = False
    
    # Artifact files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    if set(request.artifactFiles) == set(required_files):
        if len(request.artifactFiles) == len(set(request.artifactFiles)):
            adapter_files = utf8_byte_sort(request.artifactFiles)
        else:
            reason_codes.append("ADAPTER_FILE_SET")
            peft_config_pass = False
    else:
        reason_codes.append("ADAPTER_FILE_SET")
        peft_config_pass = False
    
    # Check if any file indicates full model
    for f in request.artifactFiles:
        if "model" in f.lower() and "adapter" not in f.lower():
            reason_codes.append("FULL_MODEL_ARTIFACT")
            peft_config_pass = False
            break
    
    # Checkpoint
    checkpoint_fields = ["model", "optimizer", "scheduler", "step", "rng", "dataPosition"]
    for field in checkpoint_fields:
        if not hasattr(request.checkpoint, field) or getattr(request.checkpoint, field) is None:
            reason_codes.append("INCOMPLETE_CHECKPOINT")
            checkpoint_complete = False
            break
    
    # Lineage
    if not re.match(r'^[a-f0-9]{40}$', request.baseRevision):
        reason_codes.append("MUTABLE_BASE_REVISION")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', request.datasetDigest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', request.codeDigest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', request.configDigest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if request.expectedDigests.get("datasetDigest") != request.datasetDigest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if request.expectedDigests.get("codeDigest") != request.codeDigest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if request.expectedDigests.get("configDigest") != request.configDigest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    # Effective batch
    if not is_safe_integer(request.microBatch) or request.microBatch <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(request.gradientAccumulation) or request.gradientAccumulation <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(request.replicas) or request.replicas <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(request.expectedEffectiveBatch) or request.expectedEffectiveBatch <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if (request.microBatch * request.gradientAccumulation * request.replicas != 
        request.expectedEffectiveBatch):
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    # Resume determinism
    if not request.uninterruptedWeights or not request.resumedWeights:
        reason_codes.append("RESUME_DIVERGENCE")
        resume_pass = False
    elif len(request.uninterruptedWeights) != len(request.resumedWeights):
        reason_codes.append("RESUME_DIVERGENCE")
        resume_pass = False
    else:
        if not is_safe_float(request.resumeTolerance) or request.resumeTolerance < 0:
            reason_codes.append("RESUME_DIVERGENCE")
            resume_pass = False
        else:
            for u, r in zip(request.uninterruptedWeights, request.resumedWeights):
                if not is_safe_float(u) or not is_safe_float(r):
                    reason_codes.append("RESUME_DIVERGENCE")
                    resume_pass = False
                    break
                if abs(u - r) > request.resumeTolerance:
                    reason_codes.append("RESUME_DIVERGENCE")
                    resume_pass = False
                    break
    
    # Deduplicate and sort reason codes
    reason_codes = sorted(list(set(reason_codes)), key=lambda x: x.encode('utf-8'))
    
    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": peft_config_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": reason_codes
    }


@app.post("/adapt")
async def adapt(request: Dict[str, Any]) -> Response:
    """Main endpoint handling both choose and repair operations"""
    try:
        operation = request.get("operation")
        
        if operation == "choose":
            choose_request = ChooseRequest(**request)
            return handle_choose(choose_request)
        elif operation == "repair":
            repair_request = RepairRequest(**request)
            return handle_repair(repair_request)
        else:
            return Response(content='{"error": "INVALID_INPUT"}', status_code=400, media_type="application/json")
    except Exception as e:
        # Return 400 for any validation or missing operation errors
        return Response(content='{"error": "INVALID_INPUT"}', status_code=400, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
