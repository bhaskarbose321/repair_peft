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


def handle_choose_dict(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle choose operation with dict-based validation (more lenient)"""
    priority_order = ["prompt_only", "retrieval", "lora", "qlora"]
    
    # Extract policy with defaults
    policy = request.get("policy", {})
    min_quality = policy.get("minQuality", 0.0)
    freshness_required = policy.get("freshnessRequired", False)
    max_latency = policy.get("maxLatencyMs", 0)
    max_memory = policy.get("maxMemoryMb", 0)
    max_labeled = policy.get("maxLabeledExamples", 0)
    max_cost = policy.get("maxTotalCost", 0.0)
    horizon = policy.get("horizonRequests", 0)
    
    total_costs = {}
    reason_codes = {}
    eligible = []
    
    for name in priority_order:
        candidate = None
        for c in request.get("candidates", []):
            if c.get("name") == name:
                candidate = c
                break
        
        if candidate is None:
            reason_codes[name] = ["INVALID_INPUT"]
            total_costs[name] = 0.0
            continue
        
        # Calculate total cost
        one_time = candidate.get("oneTimeCost", 0.0)
        recurring = candidate.get("recurringCost", 0.0)
        total_cost = round_to_12_decimals(one_time + horizon * recurring)
        total_costs[name] = total_cost
        
        # Validate candidate
        reasons = []
        
        if not candidate.get("available", False):
            reasons.append("UNAVAILABLE")
        
        if candidate.get("quality", 0.0) < min_quality:
            reasons.append("QUALITY_FLOOR")
        
        if freshness_required and not candidate.get("freshness", False):
            reasons.append("FRESHNESS_REQUIRED")
        
        if candidate.get("latencyMs", 0) > max_latency:
            reasons.append("LATENCY_LIMIT")
        
        if candidate.get("memoryMb", 0) > max_memory:
            reasons.append("MEMORY_LIMIT")
        
        if candidate.get("labeledExamples", 0) > max_labeled:
            reasons.append("DATA_LIMIT")
        
        if total_cost > max_cost:
            reasons.append("COST_LIMIT")
        
        reason_codes[name] = sorted(list(set(reasons)), key=lambda x: x.encode('utf-8'))
        
        if not reasons:
            eligible.append(name)
    
    # Keep eligible in priority order
    selected = eligible[0] if eligible else None
    
    return {
        "selected": selected,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes
    }


def handle_repair_dict(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle repair operation with dict-based validation (more lenient)"""
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
    tokens = request.get("tokens", [])
    if not tokens:
        reason_codes.append("INVALID_TOKEN")
        labels = [-100] * len(tokens)
    else:
        all_valid = True
        for token in tokens:
            token_id = token.get("id", -1)
            role = token.get("role", "")
            padding = token.get("padding", False)
            
            if not isinstance(token_id, int) or token_id < 0:
                all_valid = False
                break
            if role not in ["system", "user", "assistant"]:
                all_valid = False
                break
            if not isinstance(padding, bool):
                all_valid = False
                break
        
        if not all_valid:
            labels = [-100] * len(tokens)
            reason_codes.append("INVALID_TOKEN")
        else:
            for token in tokens:
                if token.get("role") == "assistant" and not token.get("padding", False):
                    labels.append(token.get("id", -100))
                else:
                    labels.append(-100)
    
    # Template applications
    if request.get("templateApplications", 0) != 1:
        reason_codes.append("CHAT_TEMPLATE_COUNT")
        template_pass = False
    
    # Parameter validation
    parameters = request.get("parameters", [])
    param_names = [p.get("name", "") for p in parameters]
    if len(param_names) != len(set(param_names)):
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    else:
        for param in parameters:
            numel = param.get("numel", 0)
            if not isinstance(numel, int) or numel <= 0:
                reason_codes.append("INVALID_PARAMETER")
                peft_config_pass = False
                break
    
    # Inference mode
    if request.get("inferenceMode", False):
        reason_codes.append("INFERENCE_MODE")
        evaluation_deterministic = False
    
    # Allowed targets
    allowed_targets = request.get("allowedTargets", [])
    if not allowed_targets:
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    elif len(allowed_targets) != len(set(allowed_targets)):
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    
    # Check for LoRA parameters
    has_lora = False
    for param in parameters:
        target = param.get("target", "")
        name = param.get("name", "")
        if target in allowed_targets:
            if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
                has_lora = True
                break
    
    if not has_lora and parameters:
        reason_codes.append("INVALID_PARAMETER")
        peft_config_pass = False
    
    # Trainable parameters
    if peft_config_pass:
        for param in parameters:
            target = param.get("target", "")
            name = param.get("name", "")
            if target in allowed_targets:
                if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
                    trainable_params.append(name)
        
        trainable_params = utf8_byte_sort(trainable_params)
        trainable_count = sum(
            p.get("numel", 0) for p in parameters 
            if p.get("name") in trainable_params
        )
    
    # Train/eval row IDs
    train_row_ids = request.get("trainRowIds", [])
    eval_row_ids = request.get("evalRowIds", [])
    
    if not train_row_ids or len(train_row_ids) != len(set(train_row_ids)):
        reason_codes.append("INVALID_PARAMETER")
        eval_isolated = False
    
    if not eval_row_ids or len(eval_row_ids) != len(set(eval_row_ids)):
        reason_codes.append("INVALID_PARAMETER")
        eval_isolated = False
    
    if set(train_row_ids) & set(eval_row_ids):
        reason_codes.append("EVAL_LEAKAGE")
        eval_isolated = False
    
    # Dropout during eval
    if request.get("dropoutActiveDuringEval", False):
        reason_codes.append("EVAL_DROPOUT_ACTIVE")
        evaluation_deterministic = False
    
    # Artifact files
    artifact_files = request.get("artifactFiles", [])
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    if set(artifact_files) == set(required_files):
        if len(artifact_files) == len(set(artifact_files)):
            adapter_files = utf8_byte_sort(artifact_files)
        else:
            reason_codes.append("ADAPTER_FILE_SET")
            peft_config_pass = False
    else:
        reason_codes.append("ADAPTER_FILE_SET")
        peft_config_pass = False
    
    # Check if any file indicates full model
    for f in artifact_files:
        if "model" in f.lower() and "adapter" not in f.lower():
            reason_codes.append("FULL_MODEL_ARTIFACT")
            peft_config_pass = False
            break
    
    # Checkpoint
    checkpoint = request.get("checkpoint", {})
    checkpoint_fields = ["model", "optimizer", "scheduler", "step", "rng", "dataPosition"]
    for field in checkpoint_fields:
        if field not in checkpoint or checkpoint[field] is None:
            reason_codes.append("INCOMPLETE_CHECKPOINT")
            checkpoint_complete = False
            break
    
    # Lineage
    base_revision = request.get("baseRevision", "")
    dataset_digest = request.get("datasetDigest", "")
    code_digest = request.get("codeDigest", "")
    config_digest = request.get("configDigest", "")
    expected_digests = request.get("expectedDigests", {})
    
    if not re.match(r'^[a-f0-9]{40}$', base_revision):
        reason_codes.append("MUTABLE_BASE_REVISION")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', dataset_digest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', code_digest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if not re.match(r'^[a-f0-9]{64}$', config_digest):
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if expected_digests.get("datasetDigest") != dataset_digest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if expected_digests.get("codeDigest") != code_digest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    if expected_digests.get("configDigest") != config_digest:
        reason_codes.append("LINEAGE_MISMATCH")
        lineage_pass = False
    
    # Effective batch
    micro_batch = request.get("microBatch", 0)
    gradient_accumulation = request.get("gradientAccumulation", 0)
    replicas = request.get("replicas", 0)
    expected_effective_batch = request.get("expectedEffectiveBatch", 0)
    
    if not is_safe_integer(micro_batch) or micro_batch <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(gradient_accumulation) or gradient_accumulation <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(replicas) or replicas <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if not is_safe_integer(expected_effective_batch) or expected_effective_batch <= 0:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    if micro_batch * gradient_accumulation * replicas != expected_effective_batch:
        reason_codes.append("EFFECTIVE_BATCH_MISMATCH")
    
    # Resume determinism
    uninterrupted_weights = request.get("uninterruptedWeights", [])
    resumed_weights = request.get("resumedWeights", [])
    resume_tolerance = request.get("resumeTolerance", 0.0)
    
    if not uninterrupted_weights or not resumed_weights:
        reason_codes.append("RESUME_DIVERGENCE")
        resume_pass = False
    elif len(uninterrupted_weights) != len(resumed_weights):
        reason_codes.append("RESUME_DIVERGENCE")
        resume_pass = False
    else:
        if not is_safe_float(resume_tolerance) or resume_tolerance < 0:
            reason_codes.append("RESUME_DIVERGENCE")
            resume_pass = False
        else:
            for u, r in zip(uninterrupted_weights, resumed_weights):
                if not is_safe_float(u) or not is_safe_float(r):
                    reason_codes.append("RESUME_DIVERGENCE")
                    resume_pass = False
                    break
                if abs(u - r) > resume_tolerance:
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
            # Use direct dict handling instead of strict Pydantic validation
            return handle_choose_dict(request)
        elif operation == "repair":
            # Use direct dict handling instead of strict Pydantic validation
            return handle_repair_dict(request)
        else:
            return Response(content='{"error": "INVALID_INPUT"}', status_code=400, media_type="application/json")
    except Exception as e:
        # Return 400 for any errors
        return Response(content='{"error": "INVALID_INPUT"}', status_code=400, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
