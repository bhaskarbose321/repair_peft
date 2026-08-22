import pytest
from fastapi.testclient import TestClient
from main import app, calculate_total_cost, round_to_12_decimals, utf8_byte_sort

client = TestClient(app)


class TestChooseOperation:
    def test_valid_choose_request(self):
        """Test valid choose request with eligible candidates"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                },
                {
                    "name": "retrieval",
                    "available": True,
                    "quality": 0.9,
                    "freshness": True,
                    "latencyMs": 80,
                    "memoryMb": 512,
                    "labeledExamples": 50,
                    "oneTimeCost": 20,
                    "recurringCost": 0.02
                },
                {
                    "name": "lora",
                    "available": True,
                    "quality": 0.95,
                    "freshness": True,
                    "latencyMs": 90,
                    "memoryMb": 768,
                    "labeledExamples": 80,
                    "oneTimeCost": 50,
                    "recurringCost": 0.05
                },
                {
                    "name": "qlora",
                    "available": True,
                    "quality": 0.88,
                    "freshness": True,
                    "latencyMs": 70,
                    "memoryMb": 384,
                    "labeledExamples": 30,
                    "oneTimeCost": 30,
                    "recurringCost": 0.03
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        # Should select the first eligible in priority order
        assert data["selected"] == "prompt_only"
        # Eligible should be in priority order
        assert data["eligible"] == ["prompt_only", "retrieval", "lora", "qlora"]
        assert "totalCosts" in data
        assert "reasonCodes" in data

    def test_unavailable_candidate(self):
        """Test candidate with available=false"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": False,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "UNAVAILABLE" in data["reasonCodes"]["prompt_only"]

    def test_quality_failure(self):
        """Test candidate with quality below minQuality"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.9,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "QUALITY_FLOOR" in data["reasonCodes"]["prompt_only"]

    def test_freshness_failure(self):
        """Test candidate with freshness=false when required"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": False,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "FRESHNESS_REQUIRED" in data["reasonCodes"]["prompt_only"]

    def test_latency_failure(self):
        """Test candidate with latency above max"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 50,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 100,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "LATENCY_LIMIT" in data["reasonCodes"]["prompt_only"]

    def test_memory_failure(self):
        """Test candidate with memory above max"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 128,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "MEMORY_LIMIT" in data["reasonCodes"]["prompt_only"]

    def test_data_limit_failure(self):
        """Test candidate with labeledExamples above max"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 10,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 50,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "DATA_LIMIT" in data["reasonCodes"]["prompt_only"]

    def test_cost_limit_failure(self):
        """Test candidate with total cost above max"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 50,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": True,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "COST_LIMIT" in data["reasonCodes"]["prompt_only"]

    def test_priority_selection(self):
        """Test that priority order is respected"""
        request = {
            "operation": "choose",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": [
                {
                    "name": "prompt_only",
                    "available": False,
                    "quality": 0.85,
                    "freshness": True,
                    "latencyMs": 50,
                    "memoryMb": 256,
                    "labeledExamples": 0,
                    "oneTimeCost": 10,
                    "recurringCost": 0.01
                },
                {
                    "name": "retrieval",
                    "available": True,
                    "quality": 0.9,
                    "freshness": True,
                    "latencyMs": 80,
                    "memoryMb": 512,
                    "labeledExamples": 50,
                    "oneTimeCost": 20,
                    "recurringCost": 0.02
                }
            ]
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["selected"] == "retrieval"

    def test_cost_rounding(self):
        """Test that total cost is rounded to 12 decimal places"""
        total = calculate_total_cost(10.123456789012345, 10000, 0.01)
        expected = 10.123456789012 + 10000 * 0.01
        expected = round_to_12_decimals(expected)
        assert total == expected
        assert len(str(total).split('.')[-1]) <= 12

    def test_invalid_operation(self):
        """Test invalid operation returns 400"""
        request = {
            "operation": "invalid",
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": []
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 400
        import json
        assert json.loads(response.content) == {"error": "INVALID_INPUT"}

    def test_missing_operation(self):
        """Test missing operation returns 400"""
        request = {
            "policy": {
                "minQuality": 0.8,
                "freshnessRequired": True,
                "maxLatencyMs": 100,
                "maxMemoryMb": 1024,
                "maxLabeledExamples": 100,
                "maxTotalCost": 1000,
                "horizonRequests": 10000
            },
            "candidates": []
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 400
        import json
        assert json.loads(response.content) == {"error": "INVALID_INPUT"}


class TestRepairOperation:
    def test_valid_repair_request(self):
        """Test valid repair request"""
        request = {
            "operation": "repair",
            "tokens": [
                {"id": 0, "role": "system", "padding": False, "text": "You are a helpful assistant."},
                {"id": 1, "role": "user", "padding": False, "text": "Hello"},
                {"id": 42, "role": "assistant", "padding": False, "text": "Hi there!"}
            ],
            "templateApplications": 1,
            "parameters": [
                {"name": "model.layers.0.lora_A.weight", "target": "model.layers.0", "numel": 1000},
                {"name": "model.layers.0.lora_B.weight", "target": "model.layers.0", "numel": 2000}
            ],
            "allowedTargets": ["model.layers.0", "model.layers.1"],
            "inferenceMode": False,
            "trainRowIds": ["row1", "row2"],
            "evalRowIds": ["row3", "row4"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0, 2.0, 3.0],
            "resumedWeights": [1.0, 2.0, 3.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == [-100, -100, 42]
        assert data["templatePass"] == True
        assert len(data["trainableParams"]) == 2
        assert data["trainableCount"] == 3000
        assert data["peftConfigPass"] == True
        assert data["checkpointComplete"] == True
        assert data["lineagePass"] == True
        assert data["evalIsolated"] == True
        assert data["evaluationDeterministic"] == True
        assert data["resumePass"] == True

    def test_invalid_token(self):
        """Test invalid token returns -100 labels"""
        request = {
            "operation": "repair",
            "tokens": [
                {"id": -1, "role": "system", "padding": False, "text": "Invalid ID"}
            ],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == [-100]
        assert "INVALID_TOKEN" in data["reasonCodes"]

    def test_padded_assistant_token(self):
        """Test padded assistant token gets -100 label"""
        request = {
            "operation": "repair",
            "tokens": [
                {"id": 42, "role": "assistant", "padding": True, "text": "Padded"}
            ],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == [-100]

    def test_multiple_assistant_tokens(self):
        """Test multiple assistant tokens"""
        request = {
            "operation": "repair",
            "tokens": [
                {"id": 0, "role": "system", "padding": False, "text": "System"},
                {"id": 42, "role": "assistant", "padding": False, "text": "First"},
                {"id": 43, "role": "assistant", "padding": False, "text": "Second"}
            ],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["labels"] == [-100, 42, 43]

    def test_invalid_parameter(self):
        """Test invalid parameter"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [
                {"name": "param1", "target": "target1", "numel": -1}
            ],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_PARAMETER" in data["reasonCodes"]
        assert data["peftConfigPass"] == False

    def test_missing_lora_parameters(self):
        """Test missing LoRA parameters"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [
                {"name": "param1", "target": "target1", "numel": 1000}
            ],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_PARAMETER" in data["reasonCodes"]

    def test_invalid_allowed_targets(self):
        """Test invalid allowedTargets"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": [],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "INVALID_PARAMETER" in data["reasonCodes"]

    def test_inference_mode_true(self):
        """Test inferenceMode=true"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": True,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "INFERENCE_MODE" in data["reasonCodes"]
        assert data["evaluationDeterministic"] == False

    def test_template_applications_not_one(self):
        """Test templateApplications != 1"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 2,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "CHAT_TEMPLATE_COUNT" in data["reasonCodes"]
        assert data["templatePass"] == False

    def test_full_model_artifact(self):
        """Test full-model artifact"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "FULL_MODEL_ARTIFACT" in data["reasonCodes"]

    def test_incorrect_adapter_file_set(self):
        """Test incorrect adapter file set"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "ADAPTER_FILE_SET" in data["reasonCodes"]

    def test_incomplete_checkpoint(self):
        """Test incomplete checkpoint"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": None
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "INCOMPLETE_CHECKPOINT" in data["reasonCodes"]
        assert data["checkpointComplete"] == False

    def test_mutable_base_revision(self):
        """Test mutable/invalid base revision"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "main",
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "MUTABLE_BASE_REVISION" in data["reasonCodes"]
        assert data["lineagePass"] == False

    def test_lineage_mismatch(self):
        """Test lineage mismatch"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "x" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "LINEAGE_MISMATCH" in data["reasonCodes"]
        assert data["lineagePass"] == False

    def test_effective_batch_mismatch(self):
        """Test effective batch mismatch"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 2,
            "gradientAccumulation": 2,
            "replicas": 2,
            "expectedEffectiveBatch": 5,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "EFFECTIVE_BATCH_MISMATCH" in data["reasonCodes"]

    def test_train_eval_overlap(self):
        """Test train/eval overlap"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1", "row2"],
            "evalRowIds": ["row2", "row3"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "EVAL_LEAKAGE" in data["reasonCodes"]
        assert data["evalIsolated"] == False

    def test_eval_dropout_active(self):
        """Test eval dropout active"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": True,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0],
            "resumedWeights": [1.0],
            "resumeTolerance": 0.0
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "EVAL_DROPOUT_ACTIVE" in data["reasonCodes"]
        assert data["evaluationDeterministic"] == False

    def test_resume_divergence(self):
        """Test resume divergence"""
        request = {
            "operation": "repair",
            "tokens": [{"id": 0, "role": "system", "padding": False, "text": "System"}],
            "templateApplications": 1,
            "parameters": [],
            "allowedTargets": ["target1"],
            "inferenceMode": False,
            "trainRowIds": ["row1"],
            "evalRowIds": ["row2"],
            "dropoutActiveDuringEval": False,
            "artifactFiles": ["adapter_config.json", "adapter_model.safetensors"],
            "baseRevision": "a" * 40,
            "datasetDigest": "b" * 64,
            "codeDigest": "c" * 64,
            "configDigest": "d" * 64,
            "expectedDigests": {
                "datasetDigest": "b" * 64,
                "codeDigest": "c" * 64,
                "configDigest": "d" * 64
            },
            "microBatch": 1,
            "gradientAccumulation": 1,
            "replicas": 1,
            "expectedEffectiveBatch": 1,
            "checkpoint": {
                "model": {},
                "optimizer": {},
                "scheduler": {},
                "step": 0,
                "rng": {},
                "dataPosition": 0
            },
            "uninterruptedWeights": [1.0, 2.0, 3.0],
            "resumedWeights": [1.1, 2.0, 3.0],
            "resumeTolerance": 0.05
        }
        response = client.post("/adapt", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "RESUME_DIVERGENCE" in data["reasonCodes"]
        assert data["resumePass"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
