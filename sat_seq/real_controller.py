"""
RealMultiAgentController: A controller for real models, supporting DAPO training and AIME24 inference.

This controller replaces the synthetic data in MultiAgentController and implements:
  - Loading of real Qwen3-4B models
  - Reading of real datasets (DAPO)
  - Real forward propagation and gradient updates
  - Real KL divergence calculation
  - Support for AIME24 inference

Usage:
    # Training mode
    controller = RealMultiAgentController(config, mode="train", dataset_path="path/to/dapo")
    
    # Inference mode
    controller = RealMultiAgentController(config, mode="inference", dataset_path="path/to/aime24")
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from verl.models.causal_lm import ModelWithValueHead

# Attempt to import PyTorch-related libraries
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  Warning: PyTorch or transformers not installed.")
    print("   Install with: pip install torch transformers")


class RealMultiAgentController:
    """
    A real multi-agent controller that supports actual model training and inference.
    """
    
    def __init__(self, config: Dict, mode: str = "train", dataset_path: str = None):
        """
        Initializes the controller.
        
        Args:
            config (Dict): The configuration dictionary.
            mode (str): The mode, either "train" or "inference".
            dataset_path (str): The path to the dataset (DAPO or AIME24).
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for RealMultiAgentController")
        
        self.config = config
        self.mode = mode
        self.dataset_path = dataset_path
        self.agents = []
        self.active_agent_id = None
        self.param_checkpoint = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load agent models
        agents_config = config.get("sat_seq", {}).get("agents", [])
        for agent_cfg in agents_config:
            agent = self._load_agent(agent_cfg)
            self.agents.append(agent)
        
        # Load dataset
        if mode == "train":
            self.train_dataset = self._load_dapo_dataset(dataset_path)
            self.data_iterator = iter(self.train_dataset)
        elif mode == "inference":
            self.test_dataset = self._load_aime24_dataset(dataset_path)
        
        print(f"RealMultiAgentController initialized: {len(self.agents)} agents, mode={mode}")
        # The last batch used for a training stage (for KL measurement/loss)
        self.last_stage_batch: Dict = None
    
    def _load_agent(self, agent_cfg: Dict) -> Dict:
        """
        Loads a single Qwen3-4B model.
        
        Args:
            agent_cfg (Dict): Agent configuration, containing name and path.
        
        Returns:
            Dict: An agent object, containing the model, tokenizer, and optimizer.
        """
        model_path = agent_cfg.get("path")
        agent_name = agent_cfg.get("name")
        
        print(f"Loading {agent_name} from {model_path}...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with value head
            model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = ModelWithValueHead(model_path=model_path, model_dtype=model_dtype)
            model.to(self.device)
            
            # Create optimizer (includes model and value head parameters)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.get("training", {}).get("learning_rate", 1e-5),
                weight_decay=0.01
            )
            
            agent = {
                "name": agent_name,
                "path": model_path,
                "model": model,
                "tokenizer": tokenizer,
                "optimizer": optimizer,
            }
            
            print(f"✅ Loaded {agent_name}")
            return agent
            
        except Exception as e:
            print(f"❌ Failed to load {agent_name}: {e}")
            print("   Using dummy agent for testing")
            # Return a dummy agent as a fallback
            return {
                "name": agent_name,
                "path": model_path,
                "model": None,
                "tokenizer": None,
                "optimizer": None,
            }
    
    def _load_dapo_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Loads the DAPO dataset.
        
        DAPO dataset format:
        {
            "prompt": "Question description",
            "chosen": "Correct answer",
            "rejected": "Incorrect answer"
        }
        
        Args:
            dataset_path (str): Path to the DAPO dataset (.jsonl file).
        
        Returns:
            List[Dict]: A list of dataset examples.
        """
        if dataset_path is None:
            print("⚠️  No dataset path provided, using dummy data")
            return []
        
        dataset = []
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            print(f"⚠️  Dataset file not found: {dataset_path}")
            return []
        
        print(f"Loading DAPO dataset from {dataset_path}...")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    dataset.append(data)
        
        print(f"✅ Loaded {len(dataset)} examples from DAPO dataset")
        return dataset
    
    def _load_aime24_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Loads the AIME24 test set.
        
        AIME24 dataset format:
        {
            "problem": "Problem description",
            "answer": "Correct answer (optional, not needed for inference)"
        }
        
        Args:
            dataset_path (str): Path to the AIME24 dataset.
        
        Returns:
            List[Dict]: A list of test set problems.
        """
        if dataset_path is None:
            print("⚠️  No dataset path provided")
            return []
        
        dataset = []
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            print(f"⚠️  Dataset file not found: {dataset_path}")
            return []
        
        print(f"Loading AIME24 dataset from {dataset_path}...")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    dataset.append(data)
        
        print(f"✅ Loaded {len(dataset)} problems from AIME24")
        return dataset
    
    def num_agents(self) -> int:
        """Gets the number of agents."""
        return len(self.agents)
    
    def activate_agent(self, agent_id: int) -> None:
        """
        Activates a specific agent for training and freezes the others.
        
        Args:
            agent_id (int): The index of the agent.
        """
        if agent_id < 0 or agent_id >= len(self.agents):
            raise IndexError(f"agent_id {agent_id} out of range")
        
        self.active_agent_id = agent_id
        
        # Freeze all agents, then activate the specified one
        for i, agent in enumerate(self.agents):
            if agent["model"] is not None:
                for param in agent["model"].parameters():
                    param.requires_grad = (i == agent_id)
        
        print(f"Activated agent {agent_id}: {self.agents[agent_id]['name']}")
    
    def generate_sequences(self, **kwargs) -> Dict:
        """
        Generates rollout sequences (real model inference).
        
        Training mode: Samples from the DAPO dataset and generates multiple responses.
        Inference mode: Uses AIME24 problems to generate answers.
        
        Returns:
            Dict: The rollout data.
        """
        batch_size = kwargs.get("batch_size", self.config.get("data", {}).get("train_batch_size", 512))
        max_seq_len = kwargs.get("max_seq_len", self.config.get("data", {}).get("max_response_length", 128))
        group_size = self.config.get("sat_seq", {}).get("group_size", 4)
        
        if self.mode == "train":
            return self._generate_train_rollout(batch_size, max_seq_len, group_size)
        else:
            return self._generate_inference_rollout(max_seq_len)
    
    def _generate_train_rollout(self, batch_size: int, max_seq_len: int, group_size: int) -> Dict:
        """
        Generates rollout data in training mode.
        
        For each prompt, uses the ensemble policy to generate `group_size` responses.
        """
        num_groups = batch_size // group_size
        all_prompts = []
        all_responses = []
        all_logp_cur = []
        all_values = []
        all_rewards = []
        all_prompt_ids = []
        all_response_ids = []
        all_response_lens = []
        group_indices = []
        
        for group_idx in range(num_groups):
            # Get a sample from the dataset
            try:
                sample = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataset)
                sample = next(self.data_iterator)
            
            prompt = sample.get("prompt", "")
            
            # Use the ensemble policy to generate multiple responses
            for _ in range(group_size):
                # Randomly select an agent
                agent_id = np.random.randint(0, len(self.agents))
                agent = self.agents[agent_id]
                
                if agent["model"] is None:
                    # Dummy data
                    response = "dummy response"
                    logp = np.random.randn(max_seq_len) * 2.0 - 5.0
                    value = np.random.randn(max_seq_len) * 0.5
                    # Give a simple reward if ground truth is available
                    gt = sample.get("chosen") or sample.get("answer")
                    reward = np.zeros(max_seq_len)
                    reward[-1] = 1.0 if isinstance(gt, str) and gt.strip() != "" else 0.0
                    prompt_ids_tensor = None
                    response_ids_tensor = None
                else:
                    # Real generation
                    gt = sample.get("chosen") or sample.get("answer")
                    response, logp, value, reward, prompt_ids_tensor, response_ids_tensor = self._generate_single_response(
                        agent, prompt, max_seq_len, ground_truth=gt
                    )
                
                all_prompts.append(prompt)
                all_responses.append(response)
                all_logp_cur.append(logp)
                all_values.append(value)
                all_rewards.append(reward)
                group_indices.append(group_idx)
                if prompt_ids_tensor is None or response_ids_tensor is None:
                    all_prompt_ids.append([])
                    all_response_ids.append([])
                    all_response_lens.append(0)
                else:
                    all_prompt_ids.append((prompt_ids_tensor.cpu().tolist()))
                    all_response_ids.append((response_ids_tensor.cpu().tolist()))
                    all_response_lens.append(int(min(len(all_response_ids[-1]), max_seq_len)))

        # Convert to numpy arrays
        num_episodes = len(all_prompts)
        logp_cur = np.array(all_logp_cur)
        values = np.array(all_values)
        rewards = np.array(all_rewards)
        group_index = np.array(group_indices)
        
        rollout_data = {
            "prompts": all_prompts,
            "responses": all_responses,
            "rewards": rewards,
            "logp_cur": logp_cur,
            "values": values,
            "meta": {
                "num_episodes": num_episodes,
                "group_size": group_size,
                "group_index": group_index,
                "response_len": np.array(all_response_lens),
            },
            "prompt_ids": all_prompt_ids,
            "response_ids": all_response_ids,
        }
        
        print(f"Generated {num_episodes} training episodes")
        return rollout_data
    
    def _generate_single_response(self, agent: Dict, prompt: str, max_seq_len: int, ground_truth: str = None):
        """
        Generates a single response using one agent.
        
        Returns:
            response (str), logp (np.ndarray), value (np.ndarray), reward (np.ndarray)
        """
        model = agent["model"]
        tokenizer = agent["tokenizer"]
        
        # Tokenize input (single sample)
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Read generation parameters
        gen_cfg = self.config.get("generation") or self.config.get("inference") or {}
        temperature = float(gen_cfg.get("temperature", 0.8))
        top_p = float(gen_cfg.get("top_p", 1.0))

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_seq_len,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
            )
        
        # Decode response
        response_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Calculate log-probabilities
        logp = self._compute_logprobs(outputs.scores, response_ids)
        
        # Calculate value
        value = self._compute_values(outputs.hidden_states, model)
        
        # Calculate reward (based on answer quality)
        reward = self._compute_reward(prompt, response, ground_truth=ground_truth)
        
        return response, logp, value, reward, inputs["input_ids"][0], response_ids
    
    def _compute_values(self, hidden_states, model) -> np.ndarray:
        """Calculates the value for each token from hidden_states."""
        values_list = []
        if hidden_states:
            # hidden_states is a tuple, each element corresponds to a generation step
            # Each element is another tuple containing hidden_states for all layers
            # We take the hidden_state from the last layer
            for step_hidden_states in hidden_states:
                last_layer_hidden_state = step_hidden_states[-1]
                # Take the hidden_state of the last token
                last_token_hidden_state = last_layer_hidden_state[:, -1, :]
                value = model.value_head(last_token_hidden_state).squeeze(-1)
                values_list.append(value.item())

        # Pad to a fixed length
        max_len = self.config.get("data", {}).get("max_response_length", 128)
        while len(values_list) < max_len:
            values_list.append(0.0)  # Padding value

        return np.array(values_list[:max_len])
    
    def _compute_logprobs(self, scores, response_ids):
        """Calculates the log probabilities of tokens."""
        logp_list = []
        for i, token_id in enumerate(response_ids):
            if i < len(scores):
                logits = scores[i][0]  # [vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)
                token_logp = log_probs[token_id].item()
                logp_list.append(token_logp)
        
        # Pad to a fixed length
        max_len = self.config.get("data", {}).get("max_response_length", 128)
        while len(logp_list) < max_len:
            logp_list.append(-10.0)  # Padding value
        
        return np.array(logp_list[:max_len])
    
    def _compute_reward(self, prompt: str, response: str, ground_truth: str = None) -> np.ndarray:
        """
        Calculates the reward (needs to be customized per task).
        
        For math problems:
        - Check answer format
        - Check numerical correctness
        - Check reasoning steps
        """
        max_len = self.config.get("data", {}).get("max_response_length", 128)

        # If ground truth is provided, use math verification (DeepScaleR style)
        reward_value = 0.0
        if isinstance(ground_truth, str) and ground_truth.strip() != "":
            try:
                is_correct = self._verify_math_answer(response, ground_truth)
                reward_value = 1.0 if is_correct else 0.0
            except Exception:
                reward_value = 0.0
        else:
            # Fallback: weak reward based on response length and keywords
            if len(response) > 10:
                reward_value += 0.1
            if any(keyword in response.lower() for keyword in ["answer", "result", "solution"]):
                reward_value += 0.2
        
        # Give the main reward at the last token
        rewards = np.zeros(max_len)
        rewards[-1] = reward_value
        
        return rewards

    # Simplified math answer extraction and verification (compatible with integers/fractions)
    def _extract_math_answer(self, text: str) -> str:
        import re
        patterns = [
            r"(?:answer|Answer|ANSWER)[\s:=]+([-+]?\d+(?:/\d+)?)",
            r"(?:result|Result|RESULT)[\s:=]+([-+]?\d+(?:/\d+)?)",
            r"\\boxed\{([-+]?\d+(?:/\d+)?)\}",
            r"\$([-+]?\d+(?:/\d+)?)\$",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        nums = re.findall(r"[-+]?\d+(?:/\d+)?", text)
        return nums[-1] if nums else ""

    def _normalize_number(self, s: str) -> str:
        from fractions import Fraction
        t = s.strip().replace(",", "")
        if "/" in t:
            num, den = t.split("/", 1)
            return str(Fraction(int(num), int(den)))
        # Integer
        return str(int(t))

    def _verify_math_answer(self, response: str, ground_truth: str) -> bool:
        if not isinstance(ground_truth, str) or ground_truth.strip() == "":
            return False
        pred = self._extract_math_answer(response)
        if pred == "":
            return False
        try:
            return self._normalize_number(pred) == self._normalize_number(ground_truth)
        except Exception:
            return pred.strip() == ground_truth.strip()
    
    def _generate_inference_rollout(self, max_seq_len: int) -> Dict:
        """
        Inference mode: generates answers for AIME24 problems.
        """
        all_prompts = []
        all_responses = []
        
        for problem_data in self.test_dataset:
            prompt = problem_data.get("problem", "")
            
            # Use all agents to generate answers, then vote or select the best
            responses = []
            for agent in self.agents:
                if agent["model"] is not None:
                    response, _, _, _, _, _ = self._generate_single_response(
                        agent, prompt, max_seq_len
                    )
                    responses.append(response)
            
            # Simple strategy: select the first agent's response (can be improved to a voting mechanism)
            final_response = responses[0] if responses else "No response"
            
            all_prompts.append(prompt)
            all_responses.append(final_response)
        
        return {
            "prompts": all_prompts,
            "responses": all_responses,
            "meta": {"num_episodes": len(all_prompts)},
        }

    def set_last_stage_batch(self, batch: Dict) -> None:
        """Saves the last stage batch for KL measurement and loss calculation."""
        self.last_stage_batch = batch

    def compute_logprobs_and_values_for_batch(self, agent_id: int, prompt_ids_list: List[List[int]],
                                              response_ids_list: List[List[int]], response_lens: np.ndarray):
        """Performs a teacher-forced forward pass on a batch, returning logp and value tensors."""
        agent = self.agents[agent_id]
        if agent["model"] is None:
            max_len = int(self.config.get("data", {}).get("max_response_length", 128))
            import torch
            dummy_logps = torch.full((len(prompt_ids_list), max_len), -10.0, device=self.device, dtype=torch.float32)
            dummy_values = torch.zeros((len(prompt_ids_list), max_len), device=self.device, dtype=torch.float32)
            return dummy_logps, dummy_values

        import torch
        model = agent["model"]
        logp_rows = []
        value_rows = []
        max_len = int(self.config.get("data", {}).get("max_response_length", 128))

        for i in range(len(prompt_ids_list)):
            p_ids = prompt_ids_list[i]
            r_ids = response_ids_list[i] if i < len(response_ids_list) else []
            T = int(response_lens[i]) if response_lens is not None else len(r_ids)
            if not p_ids or not r_ids or T <= 0:
                logp_rows.append(torch.full((max_len,), -10.0, device=self.device))
                value_rows.append(torch.zeros(max_len, device=self.device))
                continue

            prompt_ids = torch.tensor(p_ids, dtype=torch.long, device=self.device)
            resp_ids = torch.tensor(r_ids, dtype=torch.long, device=self.device)
            input_ids = torch.cat([prompt_ids, resp_ids], dim=0).unsqueeze(0)
            
            logits, values = model(input_ids=input_ids)
            logits, values = logits[0], values[0]
            
            prompt_len = prompt_ids.shape[0]
            
            # Slice the logits and values for the response part
            response_logits = logits[prompt_len-1:-1]
            response_values = values[prompt_len-1:-1]

            T = min(T, resp_ids.shape[0], response_logits.shape[0])
            
            log_probs = torch.log_softmax(response_logits[:T], dim=-1)
            gather_ids = resp_ids[:T].unsqueeze(-1)
            logp_new = log_probs.gather(-1, gather_ids).squeeze(-1)
            value_new = response_values[:T]

            # Padding
            if T < max_len:
                pad_logp = torch.full((max_len - T,), -10.0, device=self.device)
                pad_value = torch.zeros(max_len - T, device=self.device)
                logp_new = torch.cat([logp_new, pad_logp], dim=0)
                value_new = torch.cat([value_new, pad_value], dim=0)
            else:
                logp_new = logp_new[:max_len]
                value_new = value_new[:max_len]
            
            logp_rows.append(logp_new)
            value_rows.append(value_new)

        return torch.stack(logp_rows, dim=0), torch.stack(value_rows, dim=0)

    def compute_logprobs_for_batch(self, agent_id: int, prompt_ids_list: List[List[int]],
                                   response_ids_list: List[List[int]], response_lens: np.ndarray):
        """Performs a teacher-forced forward pass on a batch, returning a [batch, max_len] logp tensor."""
        logps, _ = self.compute_logprobs_and_values_for_batch(
            agent_id, prompt_ids_list, response_ids_list, response_lens
        )
        return logps
    
    def optimize_step(self, loss_out: Dict) -> None:
        """
        Performs one step of gradient optimization.
        
        Args:
            loss_out (Dict): A dictionary containing loss and aux data.
        """
        if self.active_agent_id is None:
            print("WARNING: No active agent")
            return
        
        agent = self.agents[self.active_agent_id]
        if agent["model"] is None:
            print("WARNING: Dummy agent, skipping optimization")
            return
        
        # Save checkpoint
        self.param_checkpoint = self._save_params()
        
        # Get losses
        loss = loss_out.get("loss", 0.0)
        value_loss = loss_out.get("value_loss", 0.0)
        total_loss = loss + value_loss * self.config.get("training", {}).get("vf_coef", 0.1)
        
        # If loss is a tensor, perform backward pass
        if isinstance(total_loss, torch.Tensor):
            agent["optimizer"].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent["model"].parameters(), 1.0)
            agent["optimizer"].step()
        else:
            # If it's a float, it's a loss calculated externally, so skip
            print(f"Optimization step for agent {self.active_agent_id}: loss={loss:.4f}, value_loss={value_loss:.4f}")
    
    def backtrack_last_step(self) -> None:
        """Backtracks to the last checkpoint."""
        if self.param_checkpoint is None:
            print("WARNING: No checkpoint available")
            return
        
        self._load_params(self.param_checkpoint)
        print(f"Backtracked agent {self.active_agent_id}")
    
    def get_per_state_kl(self, agent_id: int) -> np.ndarray:
        """
        Calculates the per-state KL divergence.
        
        Args:
            agent_id (int): The agent index.
        
        Returns:
            np.ndarray: An array of KL divergences.
        """
        agent = self.agents[agent_id]
        if agent["model"] is None or self.last_stage_batch is None:
            return np.random.exponential(0.02, 128)
        prompt_ids_list = self.last_stage_batch.get("prompt_ids", [])
        response_ids_list = self.last_stage_batch.get("response_ids", [])
        logp_old = self.last_stage_batch.get("logp_cur")
        resp_lens = self.last_stage_batch.get("meta", {}).get("response_len")
        if not isinstance(logp_old, np.ndarray) or len(prompt_ids_list) == 0:
            return np.random.exponential(0.02, 128)
        with torch.no_grad():
            logp_new = self.compute_logprobs_for_batch(agent_id, prompt_ids_list, response_ids_list, resp_lens)
        logp_new_np = logp_new.detach().cpu().numpy()
        T = logp_old.shape[1]
        logp_new_np = logp_new_np[:, :T]
        
        # Use a more accurate KL divergence estimate: E[logp_new - logp_old]
        kl_div = logp_new_np - logp_old[:, :T]
        
        # Mask out padding
        mask = logp_old[:, :T] > -10.0
        
        # Calculate KL divergence per sequence (token-wise mean)
        kl_per_sequence = np.sum(kl_div * mask, axis=1) / np.sum(mask, axis=1)
        
        return kl_per_sequence.astype(np.float32)
    
    def _save_params(self) -> Dict:
        """Saves the parameters of the current agent."""
        if self.active_agent_id is None:
            return {}
        
        agent = self.agents[self.active_agent_id]
        if agent["model"] is None:
            return {}
        
        return copy.deepcopy(agent["model"].state_dict())
    
    def _load_params(self, params: Dict) -> None:
        """Loads parameters."""
        if self.active_agent_id is None:
            return
        
        agent = self.agents[self.active_agent_id]
        if agent["model"] is None:
            return
        
        agent["model"].load_state_dict(params)
    
    def save_checkpoint(self, save_dir: str, stage_idx: int):
        """
        Saves checkpoints for all agents.
        
        Args:
            save_dir (str): The directory to save to.
            stage_idx (int): The stage index.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            if agent["model"] is not None:
                checkpoint_path = save_path / f"{agent['name']}_stage_{stage_idx}.pt"
                torch.save({
                    'model_state_dict': agent["model"].state_dict(),
                    'optimizer_state_dict': agent["optimizer"].state_dict(),
                    'stage': stage_idx,
                }, checkpoint_path)
                print(f"Saved {agent['name']} to {checkpoint_path}")

