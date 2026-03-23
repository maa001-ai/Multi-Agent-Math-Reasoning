import os
import sys
import threading

# Silence debugger warnings for cleaner logs
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
import subprocess
import time
import re
import ast
import tempfile
from collections import Counter
from typing import List, Dict, Any
import math
import json
import shutil
import logging
import concurrent.futures
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# ==========================================
# 🛠️ GLOBAL CONFIGURATION
# ==========================================
EVAL_MODE = True  # Set to True to run the problem test benchmark
# 🔥 AUTO-SUBMIT FIX: Automatically disable test mode during real Kaggle SUBMISSIONS
# We detect submission by checking for the official inference server in sys.modules
if 'kaggle_evaluation.aimo_3_inference_server' in sys.modules:
    EVAL_MODE = False

try:
    from z3 import *
except ImportError:
    pass # Managed in Kaggle environment via wheels

# ==========================================
# �️ INTERACTIVE LOGGING TOGGLE
# ==========================================
run_type = os.getenv('KAGGLE_KERNEL_RUN_TYPE', 'Local')
is_interactive = run_type == 'Interactive' or not os.path.exists('/kaggle')
# 🔥 LOGGING FIX: If we are in EVAL_MODE, we always want to see the logs!
VERBOSE_MODE = EVAL_MODE or not os.path.exists('/kaggle')

if is_interactive:
    # Default to verbose in Interactive mode so we can see what is happening
    VERBOSE_MODE = True
    print("\n>>> CONFIGURE LOGGING <<<")
    print("Running in Interactive Mode. Detailed logs are enabled.")
else:
    # Production/Batch mode should only be silent if we aren't benchmarking
    if not VERBOSE_MODE:
        print(f"Production Mode Detected ({run_type}). Initializing Silent Execution...")
    else:
        print(f"Benchmark Mode Detected ({run_type}). Detailed logs are enabled.")

def vprint(*args, **kwargs):
    if VERBOSE_MODE:
        print(*args, **kwargs)
    elif not kwargs.get('file'): # Don't suppress stderr if explicitly directed, but usually suppress
        pass

def patch_tokenizer(model_path):
    """
    Kaggle Fix: Patches tokenizer_config.json if it contains the broken 'TokenizersBackend' class.
    We copy ALL config/tokenizer files to a temporary writable directory to ensure a complete environment.
    """
    tmp_tokenizer_path = os.path.join("/kaggle/working", f"fixed_tokenizer_{os.path.basename(model_path)}")
    if not os.path.exists(tmp_tokenizer_path):
        os.makedirs(tmp_tokenizer_path, exist_ok=True)
    
    # Copy all files under 100MB (configs, tokenizer components, etc.)
    # This captures vocab.json, merges.txt, tokenizer.json, config.json etc.
    try:
        for f in os.listdir(model_path):
            src_file = os.path.join(model_path, f)
            if os.path.isfile(src_file) and os.path.getsize(src_file) < 100 * 1024 * 1024:
                try:
                    shutil.copy(src_file, tmp_tokenizer_path)
                except Exception: pass
    except Exception:
        # Fallback to hardcoded list if listing fails
        for f in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json", "merges.txt", "config.json"]:
            src = os.path.join(model_path, f)
            if os.path.exists(src):
                try: shutil.copy(src, tmp_tokenizer_path)
                except Exception: pass
            
    config_path = os.path.join(tmp_tokenizer_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            if data.get("tokenizer_class") == "TokenizersBackend":
                vprint(f"🔥 [CRITICAL FIX] Patching broken TokenizersBackend for {os.path.basename(model_path)}...")
                data.pop("tokenizer_class", None) 
                with open(config_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            vprint(f"Warning: Failed to patch tokenizer config: {e}")
            
    return tmp_tokenizer_path

# ==========================================
# 🚨 PHASE 2: KAGGLE OFFLINE INSTALLATION 🚨
# ==========================================
try:
    import vllm
    from vllm import LLM, SamplingParams
    print("vLLM is already installed.")
except (ImportError, RuntimeError, AttributeError) as e:
    vprint(f"vLLM import failed ({type(e).__name__}: {e}). Triggering offline installation...")
    # Kaggle mounts datasets in different locations depending on how they are attached.
    # Search all known possible mount points.
    WHEEL_SEARCH_PATHS = [
        "/kaggle/input/vllm-wheels",
        "/kaggle/input/datasets/liquidvisualsinteractive/vllm-wheels/vllm-wheels",
        "/kaggle/input/datasets/liquidvisualsinteractive/z3-solver-wheels",
        "/kaggle/input/vllm-wheels/vllm-wheels",
    ]
    wheel_dir_found = None
    for p in WHEEL_SEARCH_PATHS:
        if os.path.exists(p):
            wheel_dir_found = p
            break
    
    if wheel_dir_found:
        vprint(f"Found Wheel Directory: {wheel_dir_found}. Listing contents...")
        if VERBOSE_MODE:
            os.system(f"find {wheel_dir_found} -name '*.whl' -type f 2>/dev/null | head -30")
        
        # Try to install from the directory and all its subdirectories
        import glob
        all_whl_dirs = set([wheel_dir_found])
        for whl_file in glob.glob(os.path.join(wheel_dir_found, "**", "*.whl"), recursive=True):
            all_whl_dirs.add(os.path.dirname(whl_file))
        
        find_links = " ".join([f"--find-links={d}" for d in sorted(all_whl_dirs)])
        
        # IMPORTANT: Skip protobuf wheel! Kaggle has a working protobuf pre-installed,
        # and the bundled protobuf-6.33.5 causes 'MessageFactory' attribute errors.
        # Also skip torch to avoid conflicts with pre-installed CUDA torch.
        # IMPORTANT: Use --force-reinstall to ensure binary-compatible versions of vllm/torch/pillow
        # replace any pre-installed but incompatible Kaggle versions.
        install_cmd = f"pip install --no-index {find_links} --no-deps --force-reinstall vllm"
        vprint(f"Running (force-reinstall): {install_cmd}")
        redirect = "" if VERBOSE_MODE else " > /dev/null 2>&1"
        os.system(install_cmd + redirect)
        if not VERBOSE_MODE: print(".", end="", flush=True)
        
        # Now install vllm's dependencies, but exclude core environment packages
        # (they are already installed on Kaggle and the bundled versions often conflict)
        all_whls = sorted(glob.glob(os.path.join(wheel_dir_found, "**", "*.whl"), recursive=True))
        
        # 🔥 V12.5.12 REFINEMENT: Skip protobuf, pillow, and numpy.
        # Pillow 12.1.1 and NumPy 2.2.6 cause major ImportError mismatches with the Kaggle stack.
        skip_packages = ["protobuf", "pillow", "numpy"]
        
        # Check if we should skip torch (only if it's already working or not in wheels)
        # But if we were called, import already failed, so we likely NEED the wheels.
        
        for whl_file in all_whls:
            basename = os.path.basename(whl_file)
            if any(basename.startswith(skip) or basename.startswith(skip.replace("-", "_")) for skip in skip_packages):
                print(f"SKIPPING (pre-installed): {basename}")
                continue
            if basename.startswith("vllm"):
                continue
            redirect = "" if VERBOSE_MODE else " > /dev/null 2>&1"
            # Force reinstall core packages to avoid shadowing from pre-installed broken versions
            force_flag = "--force-reinstall" if any(p in basename.lower() for p in ["torch", "pillow", "numpy", "nvidia"]) else ""
            os.system(f"pip install --no-deps --no-index {find_links} {force_flag} {whl_file}" + redirect)
            if not VERBOSE_MODE: print(".", end="", flush=True)
        
        # 🔥 V12.5.14: CRITICAL - Invalidate module caches after binary replacement
        import importlib
        importlib.invalidate_caches()
        
        vprint("vLLM offline installation complete.")
        import vllm
        from vllm import LLM, SamplingParams
        if not VERBOSE_MODE: print("\nvLLM Installed. [COMPLETE!]")
    else:
        # Last resort: search recursively for any .whl file under /kaggle/input
        print("Searching /kaggle/input recursively for .whl files...")
        os.system("find /kaggle/input -name '*.whl' -type f 2>/dev/null | head -20")
        
        print("WARNING: No vLLM wheel directory found. Ensure vllm-wheels dataset is attached.")

import polars as pl
try:
    import kaggle_evaluation.aimo_3_inference_server
except ImportError:
    pass # Managed in main()

import jupyter_client
import atexit
import queue

# ==========================================
# PHASE 2 & 3: Stateful Jupyter Sandbox
# ==========================================
class JupyterSandbox:
    def __init__(self, kernel_manager, kernel_client):
        self.km = kernel_manager
        self.kc = kernel_client
        self.timeout = 10 
        
        # Phase 2: Memory Bounding (Anti-Crash)
        # 🔥 V6 Optimization: Adjusted to 8GB for 230GB RAM headroom + Safety guard.
        self.execute("try: import resource; resource.setrlimit(resource.RLIMIT_AS, (8000000000, 8000000000))\nexcept Exception: pass")
        
    def _fix_brackets(self, code: str) -> str:
        """🔥 V13.8 SMART STRIPPER: Strips comments, blank lines, and unmatched brackets
        to ensure 'Compressed Pure Code' as requested by the user."""
        # Strategy: scan character by character, track bracket depth,
        # skip any closing bracket that would make depth go negative.
        close_to_open = {')': '(', ']': '[', '}': '{'}
        open_brackets = set('([{')
        
        result = []
        depth = {'(': 0, '[': 0, '{': 0}
        in_string = False
        string_char = None
        in_comment = False
        escape_next = False
        
        for ch in code:
            if ch in ('\n', '\r'):
                in_comment = False
                # Collapse sequential newlines to keep it "compressed"
                if result and result[-1] not in ('\n', '\r'):
                    result.append('\n')
                continue
            if in_comment:
                continue # 🔥 V13.8: Strip the comment body
            if ch == '#' and not in_string:
                in_comment = True
                continue # 🔥 V13.8: Strip the octothorpe
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
                continue
            if ch in ('"', "'") and not in_string:
                in_string = True
                string_char = ch
                result.append(ch)
                continue
            if in_string and ch == string_char:
                in_string = False
                string_char = None
                result.append(ch)
                continue
            if in_string:
                result.append(ch)
                continue
            
            # Outside strings: track brackets
            if ch in open_brackets:
                depth[ch] += 1
                result.append(ch)
            elif ch in close_to_open:
                opener = close_to_open[ch]
                if depth[opener] > 0:
                    depth[opener] -= 1
                    result.append(ch)
                else:
                    # SKIP this extra closing bracket
                    pass
            else:
                result.append(ch)
        
        # 🧼 Cleanup trailing whitespace and empty lines
        fixed = ''.join(result).strip()
        
        # 🔥 V12.5.7b: Fix doubled braces in f-strings: {expr}} → {expr}
        fixed = re.sub(r'(\{[^{}]+\})\}', r'\1', fixed)
        
        # Verify the fix actually helps
        try:
            ast.parse(fixed)
            vprint("🔧 BRACKET FIXER: Auto-repaired unbalanced brackets.")
            return fixed
        except SyntaxError:
            # Return the fixed version anyway — it's still better than the original
            return fixed

    def extract_code(self, text: str) -> str:
        # 🔥 V12.5.3 Smart Extraction: Pick the LAST valid Python block
        if "```" in text:
            raw_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
            if raw_blocks:
                # Search backwards for the most refined block
                for potential_code in reversed(raw_blocks):
                    p_code = potential_code.strip()
                    # Python heuristic: must contain common keywords or an assignment
                    if any(k in p_code for k in ["import ", "def ", "print", "=", "sum("]):
                        # Grounding: Fix nested power ops and strip LaTeX artifacts
                        p_code = p_code.replace("$", "")
                        p_code = re.sub(r'([\w\)]+)\s*\^\s*([\w\(]+)', r'\1**\2', p_code)
                        # 🔥 V12.5.7: Auto-fix doubled brackets
                        return self._fix_brackets(p_code.strip())
                
                # Fallback to the last block if no heuristic matches
                return self._fix_brackets(raw_blocks[-1].strip())
        
        return ""

    def execute(self, code: str) -> tuple[str, str, int]:
        # Phase 3: SymPy Evaluation Forcing added to execution
        sympy_force = """
import sympy as sp
try:
    __last_val = _
    if isinstance(__last_val, sp.Basic):
        try: print(__last_val.evalf())
        except Exception: print(__last_val)
except NameError:
    pass
"""
        full_code = code + "\n" + sympy_force
        
        self.kc.execute(full_code)
        
        stdout = ""
        stderr = ""
        return_code = 0
        
        try:
            while True:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                if msg_type == 'stream':
                    if content['name'] == 'stdout':
                        stdout += content['text']
                    elif content['name'] == 'stderr':
                        stderr += content['text']
                elif msg_type == 'execute_result':
                    stdout += str(content['data'].get('text/plain', '')) + "\n"
                elif msg_type == 'error':
                    stderr += "\n".join(content['traceback'])
                    return_code = 1
                elif msg_type == 'status' and content['execution_state'] == 'idle':
                    break
                
                # Phase 5: Sandbox Resource Guard
                if len(stdout) > 5000:
                    stdout = stdout[:5000] + "\n[Output truncated due to excessive size]"
                    self.km.interrupt_kernel()
                    break
                    
        except queue.Empty:
            self.km.interrupt_kernel()
            return "", f"Error: Execution timed out after {self.timeout}s. Process killed.", 124
        except Exception as e:
            return "", f"System Error executing sandbox: {e}", 1
            
        return stdout.strip(), stderr.strip(), return_code
        
class KernelPoolManager:
    def __init__(self, pool_size: int = 4):
        self.pool = queue.Queue(maxsize=pool_size)
        self.managers = []
        for _ in range(pool_size):
            # 🔥 V14.11 Fix: Disable HistoryManager to prevent sqlite3.OperationalError: database is locked in Colab/Batch
            km = jupyter_client.KernelManager(kernel_name='python3', extra_arguments=['--HistoryManager.enabled=False'])
            km.start_kernel()
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=10)
            
            sandbox = JupyterSandbox(km, kc)
            self.pool.put(sandbox)
            self.managers.append((km, kc))
            
        atexit.register(self.cleanup)

    def get_sandbox(self) -> JupyterSandbox:
        return self.pool.get()
        
    def return_sandbox(self, sandbox: JupyterSandbox):
        self.pool.put(sandbox)

    def refresh_sandbox(self, sandbox: JupyterSandbox):
        """🔥 AgenticDLVS-Tier Fix: Completely restart kernel to prevent C-level memory leaks from SymPy/Z3."""
        vprint("Refreshing Jupyter Kernel to prevent memory bloat...")
        try:
            sandbox.kc.stop_channels()
            sandbox.km.shutdown_kernel(now=True)
            # Remove old manager from tracked list
            self.managers = [m for m in self.managers if m[0] != sandbox.km]
        except Exception as e:
            vprint(f"Warning during kernel shutdown: {e}")

        # Start a fresh kernel
        # 🔥 V14.11 Fix: Disable HistoryManager to prevent sqlite3.OperationalError: database is locked in Colab/Batch
        km = jupyter_client.KernelManager(kernel_name='python3', extra_arguments=['--HistoryManager.enabled=False'])
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=10)
        
        new_sandbox = JupyterSandbox(km, kc)
        self.managers.append((km, kc))
        self.pool.put(new_sandbox)
        
    def cleanup(self):
        vprint("Cleaning up Jupyter Kernel Pool...")
        for km, kc in self.managers:
            try:
                kc.stop_channels()
                km.shutdown_kernel(now=True)
            except Exception: pass
        self.managers = []

# ==========================================
# PHASE 3: The Winning Algorithmic Pipeline
# ==========================================
class ModelOrchestrator:
    def sanitize_7b_output(self, text: str) -> str:
        """🛡️ V14.8 THE RECON MUZZLE: Aggressive recursive cleaning & hard truncation."""
        if not text: return ""
        import re
        
        # 1. Hard Truncation (Safety first)
        if len(text) > 600:
            text = text[:600] + " ... [Truncated for brevity]"
            
        # 2. Strip non-ASCII
        text = "".join(c for c in text if ord(c) < 128)
        
        # 3. Structural Stripping (Remove hallucinated sections)
        # Remove markdown code blocks (```...```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove Glossary or Note sections
        text = re.sub(r'(Glossary|Note|Technical Details):.*', '', text, flags=re.IGNORECASE | re.DOTALL)
        # Remove prompt repeaters
        text = re.sub(r'(The task is to|Analyze this|This problem asks).*?[\.\n]', '', text, flags=re.IGNORECASE)
        # Remove word salad step markers
        text = re.sub(r'(Firstly|Secondly|Step-by-step explanation).*?[\:\n]', '', text, flags=re.IGNORECASE)

        # 4. Aggressive Phrase Blacklist
        phrase_noise = [
            r"Alright, let's tackle this problem step by step\.",
            r"First, let's understand the problem\.",
            r"The goal is to find",
            r"Let's break this down",
            r"Step by step",
            r"Alright, let's",
            r"Okay, let's",
            r"We are given",
            r"I see,\s?",
            r"Sure, here is",
            r"In this problem,",
            r"To solve this,",
            r"First, we",
            r"Next, we",
            r"Finally,",
            r"let me count",
            r"thats two",
            r"total letter counts",
            r"number of letters"
        ]
        for pattern in phrase_noise:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 5. Collapse repetitive punctuation (?, $, !, ., etc.)
        text = re.sub(r'([\?\$\!\.\,\-\=\+\*\_])\1{2,}', r'\1', text)
        
        # 6. Cleanup styling
        text = text.replace("**", "").replace("###", "").replace("##", "")
        
        # 7. Collapse excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()

    def __init__(self, model_path: str, kernel_pool: KernelPoolManager, start_time: float = None):
        """Phase 2: Offline vLLM Engine Initialized with the H100 Architecture setup."""
        import gc
        import torch
        
        self.model_path = model_path
        self.kernel_pool = kernel_pool
        self.max_iterations = 16 # 🚀 Increased from 12 to 16 for deeper reasoning
        self.start_time = start_time if start_time is not None else time.time()
        self.max_kaggle_seconds = int(4.5 * 3600)  # 4.5 hours
        self.global_time_buffer = 0  # Seconds saved from easy problems
        self.llm_lock = threading.Lock()
        self.base_time_per_problem = 300  # 🚀 Increased from 240s to 300s for Deep Thinking
        self.verified_answer = None # 🔥 V12.5.15 Cluster Coordination Flag
        
        # 🔥 V11.5 Diagnostic Library (The Loop Breaker)
        self.diagnostic_library = {
            "TimeoutError": (
                "Execution timed out (exceeded limit). Your search space is too massive. "
                "DO NOT use brute-force `for` or `while` loops for large constraints. "
                "You must mathematically reduce the problem. Use modular arithmetic, Fermat's Little Theorem, "
                "generating functions, or closed-form formulas to achieve an O(1) or O(log N) solution."
            ),
            "MemoryError": (
                "Memory Limit Exceeded. You attempted to generate a list or permutation set that is too large for RAM. "
                "Do not store massive combinations in memory. Use mathematical formulas (like `math.comb`) instead."
            ),
            "TypeError": (
                "TypeError occurred. You likely mixed SymPy symbolic objects with standard Python math functions. "
                "If using SymPy variables, DO NOT use standard `int()`, `float()`, or `math.sqrt()`. "
                "Instead, use SymPy's native methods like `sp.sqrt()`, `sp.floor()`, or `.evalf()`. ALWAYS re-import `math` and `sympy`."
            ),
            "OverflowError": (
                "OverflowError: Calculation exceeded Python's floating-point limits. "
                "Keep calculations as exact integers, or use SymPy (`sp.Rational`, `sp.Integer`). RE-IMPORT ALL LIBRARIES."
            ),
            "NotImplementedError": (
                "NotImplementedError in SymPy. The equation is too complex for `sp.solve()`. "
                "Manually simplify the expression, factor it, or solve it step-by-step. RE-IMPORT ALL LIBRARIES."
            ),
            "AssertionError": [
                "Logic Audit Failed. A mathematical constraint was violated. Review the failure, fix your logic, and ensure the code matches the problem constraints.",
                "STILL a Constraint Violation. Your code logic is inconsistent with the problem. Re-map the variables and re-verify the math before coding.",
                "CRITICAL LOGIC ERROR. The translation of the problem into code is fundamentally wrong. Stop. Rewrite the algorithm from scratch to satisfy all assertions."
            ],
            "IndentationError": [
                "Indentation Error. Python requires consistent 4-space nesting. Check your block structure.",
                "STILL Indentation. You likely mixed tabs and spaces. Rewrite with flat, standard indentation.",
                "INDENTATION LOOP. Abandon current structure. Write a fresh script with no nested functions."
            ],
            "EOF error": [
                "SyntaxError: Unexpected EOF. You have an unclosed parenthesis `(` or bracket `[`.",
                "STILL unclosed brackets. Simplify your SymPy expressions. Break long formulas into multiple lines.",
                "BRACKET COLLAPSE. Rewrite the formula step-by-step using intermediate variables."
            ],
            "SyntaxError": [
                "SyntaxError: Your code is malformed. Check for missing colons `:` or typos.",
                "STILL malformed. You are using a hallucinated operator or keyword. Write clean, standard Python.",
                "SYNTAX FAILURE. Wipe previous code. Provide a primitive version of the logic from line 1."
            ],
            "NameError": [
                "NameError: Variable or module not defined. You MUST re-import ALL libraries and redefine variables in this block.",
                "STILL NameError. Do not assume any state persisted. Write a fully self-contained script including `import sympy as sp`."
            ],
            "LogicAuditError": [
                "Logic Audit Failed: Your Python implementation was inconsistent with the problem constraints. Audit Feedback: [ERROR]. Provide a corrected Python script.",
                "STILL a Logic Violation. Your code is failing to implement the approved Constraint Mapping correctly. Re-examine the mapping and rewrite the core logic.",
                "CRITICAL IMPLEMENTATION FAILURE. Your code structure is fundamentally flawed. Rewrite the algorithm from scratch, ensuring every assertion in the audit is satisfied."
            ],
            "AttributeError": [
                "AttributeError: Method or attribute does not exist. You are likely using an outdated SymPy syntax or calling a method on a None object.",
                "STILL AttributeError. Simplify your code. Use basic arithmetic or standard SymPy classes (Point, Circle, Line) correctly. RE-IMPORT SYMPY AS SP."
            ]
        }
        
        self.system_prompt = (
            "You are an expert mathematical logician and Python programmer competing in AIMO. "
            "CRITICAL DIRECTIVES:\n"
            "1. PARENTHESIS PRECISION: You are prone to closing functions too early. You MUST ensure EVERY term of a product or sum intended for a function is INSIDE the parenthesis. \n"
            "   - BAD: `sp.sqrt(s * (s - a)) * (s - b) * (s - c)`\n"
            "   - GOOD: `sp.sqrt(s * (s - a) * (s - b) * (s - c))`\n"
            "2. SYNTAX: LIST COMPREHENSIONS. When using `sum()`, `list()`, or `any()`, ensure the generator expression is INSIDE the brackets.\n"
            "   - BAD: `sum(int(d)) for d in s` (SYNTAX ERROR)\n"
            "   - GOOD: `sum(int(d) for d in s)` (CORRECT)\n"
            "# RULE #2: SYNTAX GUARD: Block the sum(int(x)) for x in s loop. Ensure all list comprehensions have explicit brackets [sum(...)] or are wrapped in sum([...]).\n"
            "3. LITERAL VERIFICATION: You MUST manually count every character in words yourself. Do NOT trust provided counts, summaries, or string lengths from the briefing. Double-check frequencies if letters repeat.\n"
            "\n"
            "System Message Format:\n"
            "3. You are forbidden from calculating the final answer manually. Every calculation MUST be done in Python.\n"
            "4. NEVER manually count items in lists or strings. You MUST use Python's `len()` function programmatically.\n"
            "5. WORD ARRANGEMENTS: If you encounter a problem like 'KAGGLE', you are forbidden from hardcoding the number of letters. You MUST use `len(string)` or `collections.Counter(string)`.\n"
            "6. SILENT PRINTING: You MUST NOT use f-strings or conversational text in your print statements. Output ONLY the final raw numerical variable: `print(ans)`.\n"
            "7. If your final answer contains Pi or symbolic SymPy variables, you MUST convert it to a float using `.evalf()` before printing.\n"
            "8. MINIMAL CODE: Your Python code block MUST be compressed. DO NOT include comments, Docstrings, or explanatory text within the code block.\n\n"
            "PROCESS:\n"
            "1. First, you MUST use <think> </think> tags to explore the math, identify hidden traps, and verify your logic.\n"
            "2. After your <think> phase, you must output a STRICT BLUEPRINT with this header:\n"
            "   ### [PHASE 1: CONSTRAINT MAPPING]\n"
            "   (List explicit/implicit variables and targets based on your verified thoughts.)\n"
            "3. Finally, write the complete Python snippet under this header:\n"
            "   ### [PHASE 2: PYTHON IMPLEMENTATION]\n"
            "   (Write EXACTLY ONE clean, self-contained Python code block. Use print() for the final result. No boxed tagging.)"
        )
        if self.model_path and self.model_path != "mock":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("\n" + "!"*50)
                    print("CRITICAL ERROR: No GPU (H100/T4) detected!")
                    print("You must enable the 'H100 GPU' in the Kaggle 'Settings' sidebar.")
                    print("!"*50 + "\n")
                    self.llm = None
                    self.sampling_params = None
                    return # Exit early to prevent vLLM crash

                num_gpus = torch.cuda.device_count()
            except Exception as e:
                print(f"Warning during GPU detection: {e}")
                num_gpus = 0
                self.llm = None
                return
                
            try:
                from vllm.lora.request import LoRARequest # 🔥 Added LoRA Support
                self.LoRARequest = LoRARequest

                vprint(f"Loading {model_path} with vLLM (Tensor Parallel Size = {num_gpus})...")
                if not VERBOSE_MODE: print(".", end="", flush=True)

                # Check for Adapter (Flexible Path Detection)
                ADAPTER_SEARCH_PATHS = [
                    "/kaggle/input/math-lora-adapter",
                    "/kaggle/input/datasets/liquidvisualsinteractive/math-lora-adapter",
                ]
                self.adapter_path = None
                for p in ADAPTER_SEARCH_PATHS:
                    if os.path.exists(p):
                        self.adapter_path = p
                        break
                
                enable_lora = self.adapter_path is not None and "merged" not in model_path.lower()
                if enable_lora:
                    vprint(f"LoRA Adapter detected at {self.adapter_path}. Enabling LoRA...")

                self.spec_model_path = None 

                # ==========================================
                # 1. ENGINE INITIALIZATION (The H100 Split)
                # ==========================================
                # We have 80GB VRAM. We use the 14B Merged Model as Primary.
                llm_kwargs = {
                    "model": model_path,
                    "tokenizer": patch_tokenizer(model_path), 
                    "tensor_parallel_size": 1,        
                    "dtype": "bfloat16",
                    "max_model_len": 16384,            
                    "trust_remote_code": True,
                    "enable_prefix_caching": True,
                    "disable_log_stats": True,
                    "enforce_eager": False,           
                    "gpu_memory_utilization": 0.55,   # 🚀 Adjusted to 0.55 for Critic + LoRA
                    "enable_lora": enable_lora,
                    "max_loras": 1 if enable_lora else 0,
                    "disable_custom_all_reduce": True,
                }

                try:
                    self.llm = LLM(**llm_kwargs)
                except Exception as e:
                    vprint(f"Error initializing 32B model: {e}")
                    raise e

                # 🔥 AgenticDLVS-Tier: PRM Critic Loop Initialization (7B Merged Model)
                print("Loading MAA Deepseek Qwen 7B TIES Merged Critic...")
                # Search across possible mount points for the 7B Critic
                CRITIC_SEARCH_PATHS = [
                    "/kaggle/input/maa-deepseek-qwen-7b-ties-merged",
                    "/kaggle/input/datasets/liquidvisualsinteractive/maa-deepseek-qwen-7b-ties-merged",
                ]
                critic_model_path = None
                for p in CRITIC_SEARCH_PATHS:
                    if os.path.exists(p):
                        critic_model_path = p
                        break
                
                if not critic_model_path:
                    print("⚠️ WARNING: Critic model path not found. Initializing without Critic.")
                    self.critic_llm = None
                    return

                critic_kwargs = {
                    "model": critic_model_path,
                    "tokenizer": patch_tokenizer(critic_model_path),
                    "tensor_parallel_size": 1,
                    "dtype": "bfloat16",
                    "max_model_len": 4096,           
                    "trust_remote_code": True,
                    "enable_prefix_caching": True,
                    "disable_log_stats": True,
                    "enforce_eager": False,
                    "gpu_memory_utilization": 0.35,  # 🚀 Increased from 0.30 to 0.35
                    "disable_custom_all_reduce": True,
                }
                
                try:
                    self.critic_llm = LLM(**critic_kwargs)
                except Exception as e:
                    vprint(f"Error initializing 7B Critic: {e}")
                    self.critic_llm = None

                # ==========================================
                # 2. OPTIMIZED SAMPLING PARAMETERS
                # ==========================================
                self.sampling_params = SamplingParams(
                    n=1,
                    temperature=0.6,
                    top_p=0.95,
                    min_p=0.05,
                    max_tokens=8192,  # 🚀 V12.5.3: Increased from 4000 to 8192 for exhaustive reasoning
                    presence_penalty=0.1,
                    frequency_penalty=0.1,
                    repetition_penalty=1.05,
                    stop=[
                        "<|im_end|>",
                        "<|im_start|>user",
                        "<|im_start|>assistant",
                        "```output",
                        "<｜Assistant｜>",
                        "<｜User｜>",
                ],
                    logprobs=1,
                    include_stop_str_in_output=True
                )
                self.lora_request = LoRARequest("math_specialist", 1, self.adapter_path) if enable_lora else None

                # Memory Flush after init
                gc.collect()
                torch.cuda.empty_cache()

            except ImportError as e:
                print(f"CRITICAL ERROR: Failed to import vLLM during orchestrator init: {e}")
                self.llm = None
                self.sampling_params = None
        else:
            self.llm, self.sampling_params, self.critic_llm = None, None, None

    def generate(self, prompt: str) -> tuple[str, float]:
        if not self.llm: return "Mock response. \\boxed{4}", 0.0
        
        with self.llm_lock:
            # Exact Tokenizer Truncation to guarantee we never exceed max context tokens
            tokenizer = self.llm.get_tokenizer()
            tokenized_prompt = tokenizer.encode(prompt)
            
            # We must leave room for the generation max_tokens (8192).
            # We initialized the main model with max_model_len=16384 specifically to save VRAM on the H100.
            max_prompt_len = 16384 - 8192 - 100 # Safe margin (~8092 tokens)
            
            if len(tokenized_prompt) > max_prompt_len:
                # 🔥 V13.5 Brain Surgery: Keep the entire system prompt intact (800 tokens) + history
                truncated_tokens = tokenized_prompt[:800] + tokenized_prompt[-(max_prompt_len-800):]
                prompt = tokenizer.decode(truncated_tokens)
                
            outputs = self.llm.generate(
                [prompt], 
                self.sampling_params, 
                use_tqdm=False,
            )
            
            output = outputs[0].outputs[0]
            text = output.text
        
        # Phase 4: Compute Shannon Entropy
        entropy = 0.0
        if output.logprobs:
            for token_lps in output.logprobs:
                for token_id, lp_obj in token_lps.items():
                    p = math.exp(lp_obj.logprob)
                    if p > 0:
                        entropy -= p * lp_obj.logprob
            entropy = entropy / max(1, len(output.logprobs))
            
        return text, entropy

    def format_prompt(self, problem: str, previous_steps: List[Dict[str, str]] = None) -> str:
        # 🔥 V12.5.6: Clean Multi-Turn ChatML (Gemini Pro Fix)
        # Each message gets its own <|im_start|>/<|im_end|> tags so the model
        # clearly knows where its code ends and where the sandbox output begins.
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\nProblem: {problem}<|im_end|>\n"
        if previous_steps:
            for step in previous_steps:
                prompt += f"<|im_start|>{step['role']}\n{step['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def verify_with_z3(self, problem_text: str, reasoning_steps: str, candidate_ans: int, sandbox: JupyterSandbox) -> bool:
        """
        🔥 AgenticDLVS-Tier Feature: Agentic Symbolic Reasoning.
        Instead of shaky regex, we prompt the 32B model to write a formal Z3 script 
        to verify the consistency of the answer, and execute it in the sandbox.
        """
        if self.verified_answer is not None:
            return False  # Thread bailout - no need to verify!
            
        try:
            vprint(f"AgenticDLVS-Tier: Triggering Agentic Z3 Verification for candidate {candidate_ans}...")
            z3_prompt = (
                f"Problem: {problem_text}\n"
                f"Reasoning Trace: {reasoning_steps}\n"
                f"Candidate Answer: {candidate_ans}\n\n"
                "Task: Write a concise Python script using 'z3-solver' to formally verify if the candidate answer is logically consistent with the constraints. "
                "Define your variables, add constraints, and check satisfiability. "
                "Output 'Z3_SAT' if the answer is verified, or 'Z3_UNSAT' if it fails. "
                "Wrap code in ```python blocks."
            )
            # Use the main LLM to generate the proof script
            proof_response, _ = self.generate(z3_prompt)
            proof_code = sandbox.extract_code(proof_response)
            if proof_code:
                stdout, _, return_code = sandbox.execute(proof_code)
                if return_code == 0 and "Z3_SAT" in stdout.upper() and "Z3_UNSAT" not in stdout.upper():
                    vprint("AgenticDLVS-Tier Z3 VERIFIED: Formal logic proof successful.")
                    return True
            return False
        except Exception as e:
            vprint(f"Warning: Agentic Z3 failed or was inconclusive: {e}")
            return False # 🔥 SAFTEY FIX: Do not grant bonus for failed code

    def solve_trajectory(self, problem_text: str, deadline: float, initial_messages: List[Dict[str, str]] = None) -> dict:
        """Runs one full generative trajectory, optionally resuming from initial_messages."""
        messages = initial_messages.copy() if initial_messages else []
        is_clean_run = True if not initial_messages else all(m.get('role') != 'user' or 'FAILED' not in m.get('content', '') for m in initial_messages)
        total_entropy = 0.0
        steps = 0
        last_sandbox_result = None
        diagnostic_count = 0 # 🔥 AgenticDLVS-Tier: 3-repair limit
        
        # Get a persistent Jupyter Sandbox instance from the pool for this entire trajectory
        sandbox = self.kernel_pool.get_sandbox()
        
        error_tracker = {}
        best_sandbox_ans = None  # 🔥 V12.5.8: Track best answer across iterations
        
        try:
            start_iter = len([m for m in messages if m['role'] == 'assistant'])
            for iteration in range(start_iter, self.max_iterations):
                # 🚦 V14.12 Hard Budgeting: Break if we exceed the calculated per-problem deadline
                if time.time() > deadline:
                    vprint(f"🛑 Trajectory Timeout: Exceeded problem budget. Returning best answer: {best_sandbox_ans}")
                    return {"answer": best_sandbox_ans, "clean": is_clean_run, "messages": messages, "entropy": total_entropy/max(1, steps), "steps": steps, "timeout": True}

                # 🚦 V12.5.16 Aggressive Bailout: Terminate before doing any work if another thread already verified the answer!
                if self.verified_answer is not None:
                    # Return quietly. The context manager will clean this thread up instantly.
                    return {"answer": best_sandbox_ans, "clean": is_clean_run, "messages": messages, "entropy": total_entropy/max(1, steps), "steps": steps}
                
                prompt = self.format_prompt(problem_text, messages)
                
                # ... [Thinking Heartbeat] ...
                stop_h = threading.Event()
                def h_func():
                    t0 = time.time()
                    while not stop_h.is_set():
                        elapsed = int(time.time() - t0)
                        # 🧬 V12.5.1 Reasoning Stages: 🗺️ -> 🖊️ -> 🧪 -> 🧱
                        if elapsed < 5: status = "Mapping Constraints 🗺️"
                        elif elapsed < 15: status = "Algebraic Prototyping 🖊️"
                        elif elapsed < 25: status = "Logic Verification 🧪"
                        else: status = "Finalizing Code 🧱"
                        
                        sys.stdout.write(f"\r   ... {status} [{elapsed}s] ... ")
                        sys.stdout.flush()
                        time.sleep(1)
                
                h_thread = threading.Thread(target=h_func)
                h_thread.daemon = True
                h_thread.start()
                
                try:
                    with self.llm_lock:
                        # 🚦 V12.5.16: We may have waited 30s for this lock. Check if another thread won while we waited!
                        if self.verified_answer is not None:
                            return {"answer": best_sandbox_ans, "clean": is_clean_run, "messages": messages, "entropy": total_entropy/max(1, steps), "steps": steps}

                        tokenizer = self.llm.get_tokenizer()
                        tokenized_prompt = tokenizer.encode(prompt)
                        max_prompt_len = 16384 - 4096 - 100
                        if len(tokenized_prompt) > max_prompt_len:
                            # 🔥 V13.5 Brain Surgery: Keep the entire system prompt intact (800 tokens) + history
                            truncated_tokens = tokenized_prompt[:800] + tokenized_prompt[-(max_prompt_len-800):]
                            prompt = tokenizer.decode(truncated_tokens)
                            
                        outputs = self.llm.generate(
                            [prompt], 
                            self.sampling_params, 
                            use_tqdm=False,
                            lora_request=self.lora_request
                        )
                finally:
                    stop_h.set()
                    h_thread.join()
                    sys.stdout.write("\n") # 🧼 Heartbeat Cleanup
                
                output = outputs[0].outputs[0]
                response = output.text
                
                # 🧠 V12.5.1 Transparency: Show model reasoning (if any)
                thought = ""
                if "<thought>" in response:
                    thought = response.split("<thought>")[1].split("</thought>")[0].strip()
                elif "thought" in response.lower(): # Fallback for some R1 variants
                    thought = response[:200].strip()

                if thought:
                    v_thought = (thought[:150] + "...") if len(thought) > 150 else thought
                    vprint(f"💭 Reasoning: {v_thought}")

                # 🧼 V12.5.6: Sanitize response before storing
                # Strip stop tokens that got included due to include_stop_str_in_output=True
                clean_response = response.strip()
                for stop_str in ["<|im_end|>", "<|im_start|>user", "<|im_start|>assistant", "```output", "｜Assistant｜", "｜User｜"]:
                    clean_response = clean_response.replace(stop_str, "").strip()

                # 🗺️ V12.6 transparency: Extract Blueprint (Constraint Map)
                blueprint = ""
                if "### [PHASE 1: CONSTRAINT MAPPING]" in clean_response:
                    bp_part = clean_response.split("### [PHASE 1: CONSTRAINT MAPPING]")[1]
                    if "### [PHASE 2: PYTHON IMPLEMENTATION]" in bp_part:
                        blueprint = bp_part.split("### [PHASE 2: PYTHON IMPLEMENTATION]")[0].strip()
                    else:
                        blueprint = bp_part.strip()[:500]
                
                if blueprint:
                    # Constraint Map hidden in submission
                    pass

                messages.append({"role": "assistant", "content": clean_response})
                steps += 1

                # Calculate entropy as early as possible
                entropy = 0.0
                if output.logprobs:
                    for token_lps in output.logprobs:
                        for token_id, lp_obj in token_lps.items():
                            p = math.exp(lp_obj.logprob)
                            if p > 0:
                                entropy -= p * lp_obj.logprob
                    entropy = entropy / max(1, len(output.logprobs))
                total_entropy += entropy

                # Calculate entropy as early as possible
                entropy = 0.0

                # Phase 2: Execution & Tiered Diagnostics
                code = sandbox.extract_code(response)
                sandbox_ans = None
                if code:
                    # Silence internal stats for submission
                    stdout, stderr, return_code = sandbox.execute(code)
                    if return_code == 0:
                        v_stdout = stdout.strip()
                        # Sandbox output hidden in submission
                        messages.append({"role": "user", "content": stdout if stdout else "Code executed successfully."})
                        nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', stdout.strip())
                        if nums:
                            try:
                                sandbox_ans = int(round(float(nums[-1].strip().rstrip('.'))))
                                if not (0 <= sandbox_ans <= 99999): sandbox_ans = None
                                # 🔥 V12.5.8: Track best answer so it's never lost
                                if sandbox_ans is not None:
                                    best_sandbox_ans = sandbox_ans
                            except: pass
                    else:
                        is_clean_run = False
                        # 🏎️ V12.5.2: Granular Error Mapping to prevent "SyntaxError Masking"
                        error_kind = "ExecutionError"
                        if "AssertionError" in stderr: error_kind = "AssertionError"
                        elif "IndentationError" in stderr: error_kind = "IndentationError"
                        elif "SyntaxError" in stderr: error_kind = "SyntaxError"
                        elif "TypeError" in stderr: error_kind = "TypeError"
                        elif "ValueError" in stderr: error_kind = "ValueError"
                        elif "ZeroDivisionError" in stderr: error_kind = "ZeroDivisionError"
                        elif "ImportError" in stderr or "ModuleNotFoundError" in stderr: error_kind = "ImportError"
                        elif "NameError" in stderr: error_kind = "NameError"
                        elif "AttributeError" in stderr: error_kind = "AttributeError"
                        elif "EOF" in stderr or "unexpected EOF" in stderr: error_kind = "EOF error"
                        elif "Memory" in stderr: error_kind = "MemoryError"
                        elif "Timeout" in stderr: error_kind = "Timeout"
                        
                        # 🕵️ V12.5.2 Transparency: Show the actual error snippet
                        v_err = ("..." + stderr[-150:]) if len(stderr) > 150 else stderr
                        vprint(f"❌ {error_kind} detected in Sandbox: {v_err.strip()}")

                        # Track count per trajectory
                        count = error_tracker.get(error_kind, 0)
                        error_tracker[error_kind] = count + 1
                        
                        tier_msgs = self.diagnostic_library.get(error_kind, ["An error occurred. Please repair the code."])
                        repair_hint = tier_msgs[min(count, len(tier_msgs)-1)]
                        
                        if count >= 5: # 🚀 Increased from 3 to 5 for V12.5.1 Robustness
                            vprint(f"PAV NOTICE: Pruning due to {error_kind} escalation limit reached.")
                            return {"answer": None, "clean": False, "reason": f"escalation_{error_kind}", "messages": messages, "steps": steps}

                        # 🧼 V12.5.4: Contextual feedback to prevent style contamination
                        error_lines = stderr.strip().split('\n')
                        clean_error = error_lines[-1] # Actual error, e.g. "SyntaxError: ..."
                        offending_code = ""
                        
                        # Try to find the exact line of code that caused the error
                        for i in range(len(error_lines)):
                            if "    " in error_lines[i] and "^" not in error_lines[i] and "File" not in error_lines[i]:
                                offending_code = error_lines[i].strip()
                                break
                        
                        feedback_msg = f"Python Error: {clean_error}\n"
                        if offending_code:
                            feedback_msg += f"Offending Line: `{offending_code}`\n"
                        feedback_msg += f"\n[Repair Hint: {repair_hint}]"
                        
                        if error_kind == "SyntaxError" and count >= 2:
                            vprint("⚠️ TIME SAFETY: Persistent SyntaxError detected. Triggering early frustration reset.")
                            # Jump to frustration handling (step 8)
                            steps = 7 

                        messages.append({"role": "user", "content": feedback_msg})
                        continue
                else:
                    vprint("⚠️ WARNING: No Python code block found in response. Requesting rethink.")
                    messages.append({"role": "user", "content": "Please provide your reasoning followed by a Python code block."})
                    continue

                # Phase 3: If we have a Sandbox result, run the Logic Audit on the TRANSLATION.
                if sandbox_ans is not None:
                    # 🚦 V12.5.15 EARLY EXIT: If another thread already verified, bail out!
                    if self.verified_answer is not None:
                        # Return our answer anyway, but skip the expensive audit
                        return {"answer": sandbox_ans, "clean": is_clean_run, "entropy": total_entropy / max(1, steps), "logic_audited": False, "verified": False, "messages": messages, "steps": steps}
                    
                    audit_passed = False
                    
                    if self.critic_llm:
                        # Logic Audit head hidden in submission
                        # 🔥 V12.6 INTERROGATOR UPGRADE: Ruthless Skepticism
                        audit_prompt = (
                            f"STRICT LANGUAGE: Output English ONLY.\n"
                            f"TELEGRAPHIC FORMAT: Use keywords and short phrases only. NO preambles. START with verdict.\n"
                            f"NO CODE: Do NOT output code blocks. NO GLOSSARY: No definitions.\n"
                            f"STRICT RULE: Your VERY FIRST LINE must be 'VERDICT: [PASS]' or 'VERDICT: [FAIL]'.\n"
                            f"If the code has hardcoded values, uses manual counting, or skips a constraint, return VERDICT: [FAIL].\n"
                            f"Problem: {problem_text}\n"
                            f"<|im_start|>user\nProblem: {problem_text}\n\nModel's Reasoning & Code:\n{clean_response}\n\nSandbox Output: {sandbox_ans}<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                        )
                        
                        try:
                            with self.llm_lock:
                                # 🚦 V12.5.16: Check if we should even bother running the 7B critic logic audit
                                if self.verified_answer is not None:
                                    return {"answer": sandbox_ans, "clean": is_clean_run, "entropy": total_entropy / max(1, steps), "logic_audited": False, "verified": False, "messages": messages, "steps": steps}
                                
                                tokenizer = self.critic_llm.get_tokenizer()
                                tokenized_prompt = tokenizer.encode(audit_prompt)
                                if len(tokenized_prompt) > 3900:
                                    audit_prompt = tokenizer.decode(tokenized_prompt[-3900:])
                                # 🕵️ V14.1 INTERROGATOR: Expanded window to 1024 tokens.
                                audit_output = self.critic_llm.generate([audit_prompt], SamplingParams(
                                    max_tokens=200, 
                                    temperature=0.0,
                                    frequency_penalty=1.5,
                                    presence_penalty=1.2
                                ), use_tqdm=False)
                                
                                if audit_output:
                                    raw_text = audit_output[0].outputs[0].text
                                    # 🔥 V12.5.8: Strip <think> tags
                                    audit_text = raw_text.split("</think>")[-1].strip() if "</think>" in raw_text else raw_text.strip()
                                    # 🛡️ V14.7 Firewall: Sanitize Auditor response
                                    audit_text = self.sanitize_7b_output(audit_text)
                                    
                                    # Audit details hidden in submission
                                    
                                    # 🔥 V14.4: More robust parsing for First-Line Verdict
                                    raw_audit = audit_text.strip()
                                    split_audit = raw_audit.split("\n", 1)
                                    first_line = split_audit[0].upper()
                                    
                                    if "VERDICT: [PASS]" in first_line:
                                        vprint("🔍 Logic Audit: [PASS]")
                                        audit_passed = True # Promotion to Elite status
                                    elif "VERDICT: [FAIL]" in first_line:
                                        vprint("🔍 Logic Audit: [FAIL] - Repairing Mapping...")
                                        audit_passed = False
                                        # On FAIL, give the model ONE chance to repair, then move on
                                        error_kind = "LogicAuditError"
                                        count = error_tracker.get(error_kind, 0)
                                        error_tracker[error_kind] = count + 1
                                        
                                        if count < 2:
                                            fail_reason = audit_text.split("[AUDIT: FAIL]")[-1].strip()[:200]
                                            messages.append({"role": "user", "content": f"Your code produced {sandbox_ans} but audit found an error: {fail_reason}\nPlease fix the code."})
                                            continue
                                    else:
                                        # Fallback to global search if not on first line (backwards compatibility)
                                        if "VERDICT: [PASS]" in raw_audit:
                                            vprint("🔍 Logic Audit: [PASS] (Non-compliant position)")
                                            audit_passed = True
                                        elif "VERDICT: [FAIL]" in raw_audit:
                                            vprint("🔍 Logic Audit: [FAIL] (Non-compliant position)")
                                            audit_passed = False
                                            # On FAIL, give the model ONE chance to repair, then move on
                                            error_kind = "LogicAuditError"
                                            count = error_tracker.get(error_kind, 0)
                                            error_tracker[error_kind] = count + 1
                                            
                                            if count < 2:
                                                fail_reason = audit_text.split("[AUDIT: FAIL]")[-1].strip()[:200]
                                                messages.append({"role": "user", "content": f"Your code produced {sandbox_ans} but audit found an error: {fail_reason}\nPlease fix the code."})
                                                continue
                                        else:
                                            vprint("⚠️ AUDIT INCONCLUSIVE. Treating as unverified.")
                                            audit_passed = False
                        except Exception as e:
                            vprint(f"Warning: Critic failed: {e}")
                    
                    # 🔥 V12.5.9 CRITICAL: Return the answer IMMEDIATELY — audit is metadata, not a gate
                    if audit_passed:
                        full_reasoning = "\n".join([m['content'] for m in messages if m['role'] == 'assistant'])
                        z3_passed = self.verify_with_z3(problem_text, full_reasoning, sandbox_ans, sandbox) if self.critic_llm else False
                        if z3_passed:
                            vprint("🔒 Z3 FORMAL PROOF CONFIRMED")
                        self.verified_answer = sandbox_ans # 🔥 Signal other threads to stop
                        return {"answer": sandbox_ans, "clean": is_clean_run, "entropy": total_entropy / max(1, steps), "logic_audited": True, "verified": True, "prm_verified": z3_passed, "messages": messages, "steps": steps}
                    else:
                        # Even without audit pass, RETURN the answer — don't keep looping!
                        vprint(f"✅ Returning result {sandbox_ans} (audit inconclusive, answer preserved)")
                        return {"answer": sandbox_ans, "clean": is_clean_run, "entropy": total_entropy / max(1, steps), "logic_audited": False, "verified": False, "messages": messages, "steps": steps}

                # Phase 4: Process Advantage Verifier (PAV) - Mid-Trajectory Pruning
                if steps >= 5:
                    avg_entropy = total_entropy / steps
                    if avg_entropy > 5.0:
                        vprint(f"PAV NOTICE: Pruning branch due to high entropy ({avg_entropy:.2f})")
                        return {"answer": None, "clean": False, "reason": "high_entropy", "messages": messages, "steps": steps}
                
                # ... [Adaptive Hints and Final Closure Check for V12] ...
                if steps >= self.max_iterations:
                    vprint("PAV NOTICE: Pruning due to step limit.")
                    # 🔥 V12.5.8: Return best answer instead of None if we found one
                    if best_sandbox_ans is not None:
                        vprint(f"💾 Returning best sandbox answer: {best_sandbox_ans}")
                        return {"answer": best_sandbox_ans, "clean": is_clean_run, "reason": "max_steps_with_answer", "messages": messages, "steps": steps, "entropy": total_entropy / max(1, steps)}
                    return {"answer": None, "clean": False, "reason": "max_steps", "messages": messages, "steps": steps}
                if steps == 8 and sandbox_ans is None:
                    vprint(f"⚠️ AGENTIC STEERING: Frustration Interrupter triggered at step {steps}. Forcing rethink.")
                    adaptive_hint = "You have taken 8 steps but have not reached a final answer. Stop calculating and rethink your approach."
                    
                    if self.critic_llm:
                        vprint("🧠 Calling 7B Diagnostician for Crisis Recon...")
                        try:
                            with self.llm_lock:
                                tokenizer = self.critic_llm.get_tokenizer()
                                full_history = "\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in messages if m['role'] in ['assistant', 'user']])
                                
                                # 🔥 V14.4: Support history slicing for large logs
                                history_tokens = tokenizer.encode(full_history)
                                spliced_history = full_history
                                if len(history_tokens) > 3000:
                                    spliced_history = "..." + tokenizer.decode(history_tokens[-3000:])
                                
                                diagnosis_prompt = (
                                    f"<|im_start|>system\nYou are a Senior AI Algorithmic Diagnostician. You are observing a Junior AI attempt to solve a math problem.\n"
                                    f"STRICT LANGUAGE: Output English ONLY.\n"
                                    f"TELEGRAPHIC FORMAT: Keywords and short phrases only. NO full sentences. NO conversational narrative.\n"
                                    f"NO CODE: Do NOT output Python or Bash code. NO GLOSSARY: Do NOT output definitions.\n"
                                    f"Identification of technical error repeating 8 times. Provide technical post-mortem and sharp command.\n"
                                    f"Format:\n"
                                    f"[CRISIS DIAGNOSIS]: (Technical post-mortem)\n"
                                    f"[TACTICAL COMMAND]: (Rigorous 1-sentence prompt for strategy shift.)\n"
                                    f"Output BOTH sections.<|im_end|>\n"
                                    f"<|im_start|>user\nProblem: {problem_text}\n\nJunior AI History:\n{spliced_history}\n<|im_end|>\n"
                                    f"<|im_start|>assistant\n"
                                )
                                
                                diag_out = self.critic_llm.generate([diagnosis_prompt], SamplingParams(
                                    max_tokens=200, 
                                    temperature=0.0,
                                    frequency_penalty=1.5,
                                    presence_penalty=1.2
                                ), use_tqdm=False)
                            
                            if diag_out and diag_out[0].outputs:
                                raw_diag = diag_out[0].outputs[0].text.strip()
                                # 🛡️ V14.7 Firewall: Sanitize Diagnostician response
                                inner_brief = raw_diag.split("</think>")[-1].strip() if "</think>" in raw_diag else raw_diag.strip()
                                inner_brief = self.sanitize_7b_output(inner_brief)
                                
                                # Extract parts
                                diag_part = "No diagnosis provided."
                                cmd_part = "Identify the error and restart."
                                
                                if "[CRISIS DIAGNOSIS]:" in inner_brief:
                                    diag_part = inner_brief.split("[CRISIS DIAGNOSIS]:")[1].split("[TACTICAL COMMAND]:")[0].strip()
                                if "[TACTICAL COMMAND]:" in inner_brief:
                                    cmd_part = inner_brief.split("[TACTICAL COMMAND]:")[1].strip()
                                
                                # Logs hidden in submission version, but vprint will handle VERBOSE_MODE
                                vprint(f"\n{'='*40}\n🚨 CRISIS RECON REPORT\n{'='*40}")
                                vprint(f"🔍 DIAGNOSIS: {diag_part}")
                                vprint(f"🎯 COMMAND: {cmd_part}\n{'='*40}\n")
                                
                                adaptive_hint = f"CRITICAL INTERVENTION: You are stuck in a dead-end approach. {cmd_part} Rethink your entire strategy from scratch."
                        except Exception as e:
                            vprint(f"Warning: Crisis Recon failed: {e}")
                            
                    messages.append({"role": "user", "content": adaptive_hint})

                    
        finally:
            # 🚀 Smart Kernel Cleanup: Only refresh if it's dirty or crashed
            if not is_clean_run or sandbox.kc is None:
                self.kernel_pool.refresh_sandbox(sandbox)
            else:
                # Lightweight reset for clean runs
                try: 
                    sandbox.execute("%reset -f")
                    self.kernel_pool.return_sandbox(sandbox) # 🔥 CRITICAL: Return to pool!
                except: 
                    self.kernel_pool.refresh_sandbox(sandbox)
            
        # 🔥 V12.5.8: Return best answer instead of None
        if best_sandbox_ans is not None:
            vprint(f"💾 Returning best sandbox answer from fallback: {best_sandbox_ans}")
            return {"answer": best_sandbox_ans, "clean": is_clean_run, "messages": messages, "entropy": total_entropy/max(1, steps), "steps": steps}
        return {"answer": None, "clean": is_clean_run, "messages": messages, "entropy": total_entropy/max(1, steps), "steps": steps}

    def solve_dynamic(self, problem_text: str) -> str:
        """Phase 3 & 4: Threaded Parallel Router with H100 Optimization."""
        
        is_tricky = False
        # 🔥 V11.9: "Vibe Expansion" (Universal Domain Routing)
        if re.search(r"triangle|circle|angle|radius|polygon|geometry|area|perimeter|coordinate|sin|cos|tan|trigonometry|arctan", problem_text, re.I):
            problem_text += (
                "\n\n[System Hint: Geometry/Trigonometry detected. Use exact sympy.geometry representations or trigonometric identities. "
                "DO NOT use float(math.pi) or math.sqrt(). Use sp.pi and sp.sqrt() for exact symbolic precision. "
                "CRITICAL: Keep the WHOLE Heron formula inside the sqrt: `sp.sqrt(s*(s-a)*(s-b)*(s-c))`.]"
            )
            vprint("🎯 Routing: Geometry Hint injected.")
            is_tricky = True
        elif re.search(r"probability|ways|arrange|permutation|combination|stars and bars|subset|contain|at least|color|coloring|graph|vertices|edges|grid|board|dice|coin|choose", problem_text, re.I):
            problem_text += (
                "\n\n[System Hint: Combinatorics/Set Theory detected. Beware of massive search spaces ($2^N$). Do not write for loops that exceed 10^5 iterations. "
                "CRITICAL: Keep both arguments INSIDE combinations: `math.comb(n, k)` or `sp.binomial(n, k)`. DO NOT write `comb(n), k`.]"
            )
            vprint("🎯 Routing: Combinatorics Hint injected.")
            is_tricky = True
        elif re.search(r"expected number|expected value|coin|flip|die|dice|probability", problem_text, re.I):
            problem_text += "\n\n[System Hint: Expected Value / Markov Chain detected. DO NOT use random simulations. Define the state transitions and set up a system of linear equations (e.g., E_0 = 1 + 0.5*E_1 + 0.5*E_0). Solve the system algebraically using sympy.solve or substitution.]"
            vprint("🎯 Routing: Markov Chain Hint injected.")
            is_tricky = True
        elif re.search(r"sequence|recurrence|series|matrix exponentiation|limit|converge", problem_text, re.I):
            problem_text += "\n\n[System Hint: Sequence/Recurrence detected. Look for closed-form formulas (e.g., r^n), characteristic equations, or matrix exponentiation to avoid O(N) loops. If the exponent is massive (like 10^9), use modular exponentiation pow(base, exp, mod) for terms in the closed-form formula.]"
            vprint("🎯 Routing: Sequence Hint injected.")
            # Sequences can be tricky but often have formulaic solutions. 
            # We'll leave it as non-tricky for now unless it's Number Theory/Combinatorics.
        elif re.search(r"remainder|divide|mod|congruent|last digit|divisor|gcd|lcm|diophantine|integer solutions|prime factor", problem_text, re.I):
            problem_text += "\n\n[System Hint: Number Theory detected. DO NOT calculate the full number if it is large. Use the Chinese Remainder Theorem, Fermat's Little Theorem, or modular exponentiation (pow(a, b, m)) to solve. For factorial sums modulo N, recall that i! = 0 (mod N) for all i >= N (or even earlier).]"
            vprint("🎯 Routing: Number Theory Hint injected.")
            is_tricky = True
        elif re.search(r"polynomial|root|coefficient|equation|degree|factor|roots|algebra|complex number|imaginary|magnitude", problem_text, re.I):
            problem_text += "\n\n[System Hint: Algebra/Polynomial detected. Use Vieta's Formulas, the Rational Root Theorem, or SymPy sp.solve() and sp.expand(). Keep all coefficients as exact fractions or integers. For complex numbers, use sp.I.]"
            vprint("🎯 Routing: Algebra Hint injected.")
            # Polynomials are baseline for LLMs.
        elif re.search(r"minimum|maximum|range|inequality|greater than|less than|AM-GM|Cauchy-Schwarz", problem_text, re.I):
            problem_text += "\n\n[System Hint: Inequality/Optimization detected. Check if the boundary case (equality) gives the solution. Apply AM-GM, Cauchy-Schwarz, or derivatives via SymPy sp.diff() and sp.solve().]"
            vprint("🎯 Routing: Inequality Hint injected.")
            is_tricky = True
        elif re.search(r"game|Alice|Bob|strategy|turn|winning|player", problem_text, re.I):
            problem_text += "\n\n[System Hint: Game Theory/Logic detected. Map out the winning states working backward from the end state (retrograde analysis). Identify invariants or parity (even/odd) conditions that force a win or tie.]"
            vprint("🎯 Routing: Game Theory Hint injected.")
            is_tricky = True

        target_samples = 32
        bonus_samples = int(self.global_time_buffer // 60)
        target_samples = min(128, target_samples + bonus_samples)
        
        vprint(f"🚀 Allocating {target_samples} parallel paths for this problem (includes {bonus_samples} bonus paths).")
        
        # ==========================================
        # 🔥 V13: The 7B Reconnaissance Agent
        # ==========================================
        problem_briefing = ""
        augmented_problem = problem_text # 🛡️ V14.10: Start with pristine problem
        if self.critic_llm:
            vprint("🕵️ 7B Recon Agent: Analyzing problem for hidden traps...")
            recon_prompt = (
                "<|im_start|>system\n"
                "You are an elite Mathematical Threat Analyst. Your job is to read an Olympiad math problem and output a STRICT, 3-bullet-point briefing for a junior coder.\n"
                "STRICT LANGUAGE: Output English ONLY.\n"
                "TELEGRAPHIC FORMAT: Use keywords and short phrases only. NO full sentences. NO conversational filler.\n"
                "NO CODE: Do NOT output code blocks. NO GLOSSARY: No definitions.\n"
                "NO MATH: DO NOT count letters. DO NOT list characters. DO NOT calculate lengths.\n"
                "STRUCTURE:\n"
                "1. Domain: [Keywords]\n"
                "2. Types: [Keywords/Ranges]\n"
                "3. Trap: [Technical Warning]\n\n"
                "EXAMPLE:\n"
                "BAD: 'The word KAGGLE has 6 letters and two Gs.'\n"
                "GOOD: '1. Domain: Combinatorics / Arranging.\\n2. Types: Identical items.\\n3. Trap: Overcounting repeating letters.'\n\n"
                "DO NOT solve. DO NOT do math. Extract parameters only.\n"
                "ARTIFACT GUARD: No LaTeX blocks. No code blocks. No glossaries.\n"
                "Output ONLY the 3 bullet points.<|im_end|>\n"
                f"<|im_start|>user\nAnalyze this problem:\n{problem_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            try:
                with self.llm_lock:
                    tokenizer = self.critic_llm.get_tokenizer()
                    tokenized = tokenizer.encode(recon_prompt)
                    # 🕵️ V14.5 SCOUT PASS: English enforcement and repetition suppression.
                    # 🕵️ V14.8 MUZZLE: Restricted to 200 tokens. NO GLOSSARY. NO CODE.
                    recon_output = self.critic_llm.generate([recon_prompt], SamplingParams(max_tokens=200, temperature=0.0, frequency_penalty=1.5, presence_penalty=1.2), use_tqdm=False)
                
                if recon_output and recon_output[0].outputs:
                    raw_brief = recon_output[0].outputs[0].text.strip()
                    # Strip <think> tags if the 7B R1 model uses them
                    briefing_text = raw_brief.split("</think>")[-1].strip() if "</think>" in raw_brief else raw_brief.strip()
                    # 🛡️ V14.7 Firewall: Final English/Repetition check
                    briefing_text = self.sanitize_7b_output(briefing_text)
                    
                    if len(briefing_text) > 10:
                        problem_briefing = f"\n\n[7B RECONNAISSANCE BRIEFING]\n{briefing_text}"
                        vprint(f"📋 Recon Briefing Acquired:\n{briefing_text}")
                        # 🛡️ V14.10: Append to augmented version only. Preserve original problem_text.
                        augmented_problem = problem_text + problem_briefing
            except Exception as e:
                vprint(f"Warning: Recon Agent failed: {e}")

        # Start standard parallel solving
        trajectories = []
        valid_seeds = []
        problem_start_time = time.time()
        dynamic_base_time = self.base_time_per_problem
        budget_available = dynamic_base_time + self.global_time_buffer
        
        # 🛡️ V14.12/14.13 Hard Timing Architecture
        deadline = problem_start_time + budget_available
        # thinking floor: 70% of the allotated base time must pass before we accept unverified majority
        min_thinking_time = problem_start_time + (dynamic_base_time * 0.7) 

        self.verified_answer = None # Reset for current problem

        def check_exit_conditions(trajs):
            """
            🔥 V12.5 Deterministic Routing:
            If the code passes the 7B Logic Audit, it is a mathematical proof. We exit immediately.
            If the audit fails but the code runs, we fall back to a high-threshold Majority Vote.
            """
            if not trajs: return None
            
            # 1. The Fast-Exit (Absolute Truth)
            # 🚀 V12.5.1: Exit only if formally verified by Z3
            verified_trajs = [t for t in trajs if t.get("prm_verified") and t["clean"]]
            if verified_trajs:
                ans = verified_trajs[0]['answer']
                vprint(f"🚀 ELITE VERIFIED: Answer {ans} mathematically proven. Terminating. (Total Solve Time: {time.time() - problem_start_time:.1f}s)")
                return ans
            
            # ⏳ V14.13 Optimal Thinking: If not verified, we must WAIT until min_thinking_time
            now = time.time()
            if now < min_thinking_time and now < deadline - 10:
                # We still have time to think. Don't settle for majority yet.
                return None

            # 2. The Fallback Consensus (Blind Majority Voting)
            # We only arrive here if the 7B Critic keeps rejecting the logic.
            valid_trajs = [t for t in trajs if t["answer"] is not None and t["clean"]]
            ans_counts = Counter([t["answer"] for t in valid_trajs])
            
            for ans, count in ans_counts.items():
                # Base threshold: require 4 independent paths
                r_c = 4  
                
                # Tricky Question Guard: Raise threshold for difficult domains
                if is_tricky:
                    r_c = 6
                
                # Buffer-Aware Precision
                if self.global_time_buffer > 1200:
                    r_c += 1
                
                # 🏎️ Clock Pressure Valve
                if (time.time() - self.start_time) > 14400: # 4 Hours
                    r_c = 2

                if count >= r_c:
                    # 🔥 V12.6: Consensus Quality Lock
                    # If the problem is tricky, we MUST have some verified seeds in the count.
                    if is_tricky:
                        verified_count = sum(1 for t in valid_trajs if t["answer"] == ans and t.get("verified"))
                        # If less than 40% of the consensus is verified, keep searching if we have time.
                        if verified_count < (r_c // 2) and self.global_time_buffer > 300:
                            vprint(f"🕵️ CONSENSUS LOCK: Answer {ans} has count {count} but only {verified_count} are verified. Continuing search...")
                            continue
                            
                    vprint(f"⚠️ FALLBACK CONSENSUS: Answer {ans} reached Unverified Threshold {r_c} (Current count: {count}). Terminating.")
                    return ans
                    
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Phase 1: Parallel Seeding 
            vprint(f"🌀 Launching Phase 1: {min(6, target_samples)} parallel seeds...")
            # 🛡️ V14.10: Ensure we send the augmented problem (Problem + Briefing) to trajectories
            seed_futures = {executor.submit(self.solve_trajectory, augmented_problem, deadline): i for i in range(min(6, target_samples))}
            
            for future in concurrent.futures.as_completed(seed_futures):
                traj = future.result()
                valid_seeds.append(traj)
                if traj["answer"] is not None:
                    trajectories.append(traj)
                    exit_ans = check_exit_conditions(trajectories)
                    if exit_ans is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return self._finalize_solve(exit_ans, problem_start_time, dynamic_base_time)

            # Phase 1.5: Verify we have something to build on
            successful_seeds = [s for s in valid_seeds if s.get("answer") is not None]
            
            # 🔥 V11.11 Aggressive Diversity Guard
            # If ONLY one seed survived and it's unverified, attempt a larger Phase 1 Refresh (6 seeds).
            if len(successful_seeds) < 2 and not any(s.get("verified") for s in successful_seeds):
                 vprint("⚠️ DIVERSITY GUARD: Only skeptical survivors. Launching AGGRESSIVE Phase 1 Refresh (6 seeds)...")
                 refresh_futures = {executor.submit(self.solve_trajectory, augmented_problem, deadline): i for i in range(6)}
                 for future in concurrent.futures.as_completed(refresh_futures):
                     if time.time() > deadline: break
                     traj = future.result()
                     valid_seeds.append(traj)
                     if traj["answer"] is not None:
                         trajectories.append(traj)
                         exit_ans = check_exit_conditions(trajectories)
                         if exit_ans is not None:
                             executor.shutdown(wait=False, cancel_futures=True)
                             return self._finalize_solve(exit_ans, problem_start_time, dynamic_base_time)
            
            successful_seeds = [s for s in valid_seeds if s.get("answer") is not None]
            verified_seeds = [s for s in successful_seeds if s.get("verified")]
            dup_source = verified_seeds if verified_seeds else successful_seeds
            
            # Phase 2: Parallel Duplication
            if dup_source and len(trajectories) < target_samples:
                remaining = target_samples - len(trajectories)
                vprint(f"🌀 Launching Phase 2: Duplicating {len(dup_source)} seeds across {remaining} parallel paths...")
                
                # 🔥 V11: Optimized Phase 2 Scoping Fix
                # Use a dictionary mapping future -> index for asynchronous completion tracking.
                dup_futures = {}
                for i in range(remaining):
                    if (time.time() - self.start_time) > self.max_kaggle_seconds or (time.time() - problem_start_time) > budget_available:
                        break
                    seed = dup_source[i % len(dup_source)]
                    dup_futures[executor.submit(self.solve_trajectory, augmented_problem, deadline, initial_messages=seed["messages"][:2])] = i

                for future in concurrent.futures.as_completed(dup_futures):
                    traj = future.result()
                    if traj["answer"] is not None:
                        trajectories.append(traj)
                        exit_ans = check_exit_conditions(trajectories)
                        if exit_ans is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return self._finalize_solve(exit_ans, problem_start_time, dynamic_base_time)

        # Phase 3: Weighted Voting
        return self._finalize_solve(self._weighted_vote(trajectories, valid_seeds), problem_start_time, dynamic_base_time)

    def _weighted_vote(self, trajectories, valid_seeds):
        if not trajectories:
            vprint("WARNING: No trajectories. Attempting Deep Consensus Extraction...")
            all_numbers = []
            for s in valid_seeds:
                if "messages" in s:
                    last_msg = [m["content"] for m in s["messages"] if m["role"] == "assistant"][-1]
                    nums = re.findall(r'\\boxed\{(\d+)\}', last_msg)
                    if nums:
                        try:
                            val = int(nums[-1])
                            if 0 <= val <= 99999:
                                all_numbers.append(val)
                        except: pass
            return Counter(all_numbers).most_common(1)[0][0] if all_numbers else "0"

        has_tier1 = any(t.get("verified") and t.get("prm_verified") for t in trajectories)
        has_tier2 = any(t.get("verified") or t.get("prm_verified") for t in trajectories)
        
        final_vote_scores = Counter()
        for t in trajectories:
            ans = t["answer"]
            if has_tier1:
                if not (t.get("verified") and t.get("prm_verified")): continue
            elif has_tier2:
                if not (t.get("verified") or t.get("prm_verified")): continue
            
            weight = 1.0 / (max(0.05, t.get("entropy", 1.0)) + 1e-9)
            if t["clean"]: weight *= 1.2
            
            # 🎯 V12.5.5 Repair Penalty: 10% penalty per repair step (Gemini Pro suggestion)
            steps_taken = t.get("steps", 1)
            weight *= (0.90 ** (steps_taken - 1))
            
            if not t.get("prm_verified"): weight *= 0.5
            if t.get("verified"): weight *= 50.0
            if t.get("prm_verified"): weight *= 15.0
            final_vote_scores[ans] += min(50.0, weight)
            
        return final_vote_scores.most_common(1)[0][0] if final_vote_scores else Counter([t["answer"] for t in trajectories]).most_common(1)[0][0]

    def _finalize_solve(self, best_ans, problem_start_time, dynamic_base_time):
        time_spent = time.time() - problem_start_time
        if time_spent < dynamic_base_time:
            saved = dynamic_base_time - time_spent
            self.global_time_buffer += saved
            vprint(f"Banked {saved:.1f}s. Total buffer: {self.global_time_buffer:.1f}s.")
        else:
            self.global_time_buffer = max(0, self.global_time_buffer - (time_spent - dynamic_base_time))
        
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        return str(best_ans)

def _fix_kaggle_weight_filenames(model_path: str) -> str:
    """
    🔥 CRITICAL FIX: Kaggle auto-renames safetensor shards during dataset upload.
    HuggingFace standard: model-00001-of-00015.safetensors
    Kaggle renames to:    model-1.safetensors
    
    This creates a writable directory with symlinks to the real weights,
    plus a patched model.safetensors.index.json that uses the actual filenames.
    """
    import glob as glob_mod
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(f"⚠️ No model.safetensors.index.json found at {model_path}. Skipping filename fix.")
        return model_path
    
    # Read the index to check what filenames it expects
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    expected_files = sorted(set(index_data.get("weight_map", {}).values()))
    actual_safetensors = sorted([
        os.path.basename(f) for f in glob_mod.glob(os.path.join(model_path, "*.safetensors"))
    ])
    
    # Check if there's actually a mismatch
    if set(expected_files) == set(actual_safetensors):
        print(f"✅ Weight filenames match index. No fix needed.")
        return model_path
    
    print(f"⚠️ Kaggle filename mismatch detected!")
    print(f"   Index expects: {expected_files[:3]}...")
    print(f"   Actual files:  {actual_safetensors[:3]}...")
    
    # Build a mapping from expected -> actual filenames
    # e.g., model-00001-of-00015.safetensors -> model-1.safetensors
    import re as re_mod
    
    # Map based on the shard number
    actual_by_num = {}
    for f in actual_safetensors:
        m = re_mod.match(r'model-(\d+)\.safetensors', f)
        if m:
            actual_by_num[int(m.group(1))] = f
    
    expected_by_num = {}
    for f in expected_files:
        m = re_mod.match(r'model-(\d+)-of-\d+\.safetensors', f)
        if m:
            expected_by_num[int(m.group(1))] = f
    
    rename_map = {}  # old_name -> new_name (actual on disk)
    for num in expected_by_num:
        if num in actual_by_num:
            rename_map[expected_by_num[num]] = actual_by_num[num]
    
    if not rename_map:
        print(f"⚠️ Could not build filename mapping. Attempting to proceed with original path.")
        return model_path
    
    # Create a writable directory with symlinks + patched index
    fixed_path = os.path.join("/kaggle/working", "fixed_model_weights")
    os.makedirs(fixed_path, exist_ok=True)
    
    # Symlink ALL files from the original directory
    for item in os.listdir(model_path):
        src = os.path.join(model_path, item)
        dst = os.path.join(fixed_path, item)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                # If symlinks fail (shouldn't on Linux), copy small files
                if os.path.getsize(src) < 100 * 1024 * 1024:  # < 100MB
                    shutil.copy(src, dst)
    
    # Now patch the index: replace expected filenames with actual filenames
    new_weight_map = {}
    for tensor_name, old_filename in index_data["weight_map"].items():
        new_weight_map[tensor_name] = rename_map.get(old_filename, old_filename)
    
    index_data["weight_map"] = new_weight_map
    
    # Write the patched index (overwrite the symlinked one)
    patched_index_path = os.path.join(fixed_path, "model.safetensors.index.json")
    if os.path.islink(patched_index_path):
        os.unlink(patched_index_path)
    with open(patched_index_path, 'w') as f:
        json.dump(index_data, f)
    
    # Verify the fix
    patched_files = sorted(set(new_weight_map.values()))
    existing_files = [f for f in patched_files if os.path.exists(os.path.join(fixed_path, f))]
    print(f"✅ Patched index: {len(patched_files)} weight files referenced, {len(existing_files)} exist on disk.")
    
    if len(existing_files) == len(patched_files):
        print(f"✅ All weight files resolved! Using fixed path: {fixed_path}")
        return fixed_path
    else:
        missing = [f for f in patched_files if f not in [os.path.basename(e) for e in existing_files]]
        print(f"❌ Still missing {len(missing)} files: {missing[:5]}")
        return fixed_path  # Try anyway

# ==========================================
# PHASE 4: Local Mock / AIMO Server Loop
# ==========================================
def main():
    import sys
    KAGGLE_MODE = "kaggle_evaluation.aimo_3_inference_server" in sys.modules
    
    # In Kaggle, datasets are mounted in different locations. Search all known paths.
    MODEL_PATH = "mock"
    import os # Make sure os is in scope for the main block evaluation
    if KAGGLE_MODE:
        # 🔥 Dynamic Search: Find any directory with config.json and prefer "merged"
        try:
            import glob
            configs = glob.glob("/kaggle/input/**/config.json", recursive=True)
            print(f"DEBUG: Found {len(configs)} config.json files: {configs}")
            # Strategy: 1. MAA Merged DS, 2. Anything else (excluding 7B critic & modernbert)
            m_ds = [os.path.dirname(c) for c in configs if "maa-deepseek-qwen-14b-ties-merged" in c.lower()]
            # 🔥 SAFETY: Exclude the 7B critic model AND modernbert from fallback to prevent silent downgrade
            m_valid = [os.path.dirname(c) for c in configs if "modernbert" not in c.lower() and "maa-deepseek-qwen-7b-ties-merged" not in c.lower() and "deepseek-r1-distill-qwen-7b" not in c.lower()]
            
            if m_ds:
                MODEL_PATH = m_ds[0]
            elif m_valid:
                MODEL_PATH = m_valid[0]
            else:
                print("⚠️ WARNING: Dynamic search found NO valid 14B model! Only 7B critic or unknown models detected.")
                print("⚠️ Please attach the 'maa-deepseek-qwen-14b-ties-merged' dataset to this notebook!")
                
            if MODEL_PATH != "mock":
                print(f"✅ Dynamic search found model weights at: {MODEL_PATH}")
        except Exception as e:
            print(f"Warning during dynamic search: {e}")

        if MODEL_PATH == "mock":
            # 🔥 FIX: Search subdirectories for the 14B model config
            MODEL_SEARCH_ROOTS = [
                "/kaggle/input/maa-deepseek-qwen-14b-ties-merged",
                "/kaggle/input/datasets/avergonzado/maa-deepseek-qwen-14b-ties-merged",
                "/kaggle/input/datasets/liquidvisualsinteractive/maa-deepseek-qwen-14b-ties-merged",
            ]
            for root in MODEL_SEARCH_ROOTS:
                if os.path.exists(root):
                    # First check if config.json is directly in this folder
                    if os.path.isfile(os.path.join(root, "config.json")):
                        MODEL_PATH = root
                        print(f"✅ Fallback found model at root: {MODEL_PATH}")
                        break
                    # Otherwise, search subdirectories for config.json
                    for dirpath, dirnames, filenames in os.walk(root):
                        if "config.json" in filenames:
                            MODEL_PATH = dirpath
                            print(f"✅ Fallback found model in subdirectory: {MODEL_PATH}")
                            break
                    if MODEL_PATH != "mock":
                        break
        
        if MODEL_PATH == "mock":
            print("❌ FATAL: Could not find any valid model with config.json!")
            print("❌ Please verify the 14B model dataset is properly attached and contains model files.")
            MODEL_PATH = "/kaggle/input/maa-deepseek-qwen-14b-ties-merged"
    
        # 🔥 CRITICAL FIX: Kaggle renames safetensor shards from HF standard names
        # (e.g., model-00001-of-00015.safetensors) to simplified names (model-1.safetensors).
        # vLLM reads model.safetensors.index.json which references the ORIGINAL names and crashes.
        # Fix: Create a writable copy with symlinks and a patched index.json.
        if MODEL_PATH != "mock":
            MODEL_PATH = _fix_kaggle_weight_filenames(MODEL_PATH)
            
    # Fast / Slow Router State
    hard_problems_queue = []
    MAX_KAGGLE_HOURS = 4.5  # GPU notebooks have 5h limit on AIMO3; 0.5h safety margin
    START_TIME = time.time()
    
    # Initialize a pool of 12 persistent Jupyter kernels
    os.environ['AIMO_PIPELINE'] = '1' # Signal to test.py to allow interactive choice
    kernel_pool = KernelPoolManager(pool_size=12)
    orchestrator = ModelOrchestrator(MODEL_PATH, kernel_pool, start_time=START_TIME)
    
    # --- CONFIGURATION FOR TESTING ---
    # Automatically uses Mock mode in Interactive, and Real Mode during Submission/Batch runs.
    is_interactive = os.getenv('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
    MOCK_KAGGLE_TEST = is_interactive  # True during editing, False during Submit
    
    if KAGGLE_MODE and EVAL_MODE:
        while True:
            vprint("\n🏆 INITIATING TEST BENCHMARK (test.py)...")
            try:
                # 🔥 FIX: Specifically avoid collision with built-in 'test' module
                import importlib.util
                import sys
                
                test_path = "/kaggle/working/test.py" if os.path.exists("/kaggle/working/test.py") else "test.py"
                if not os.path.exists(test_path):
                    vprint("⚠️ test.py not found in /kaggle/working/ or current dir.")
                    return

                spec = importlib.util.spec_from_file_location("custom_test", test_path)
                test_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_mod)
                ds = getattr(test_mod, "TEST_PROBLEMS", [])
                
                if not ds:
                    vprint("⚠️ TEST_PROBLEMS not found in test.py.")
                    return
                vprint(f"📦 Loaded {len(ds)} test problems.")
            except Exception as e:
                vprint(f"⚠️ Failed to load test.py. Error: {e}")
                return
                
            correct = 0
            total = len(ds)
            
            for i, problem in enumerate(ds):
                p_id = problem.get('id', 'ID_MISSING')
                vprint(f"\n[{i+1}/{total}] Benchmarking [{p_id}]: {problem['problem'][:150]}...")
                t_prob_start = time.time()
                ans = orchestrator.solve_dynamic(problem['problem'])
                t_prob_spent = time.time() - t_prob_start
                
                # Cleanup answer for comparison
                try:
                    pred_val = int(float(re.sub(r'[^0-9.-]', '', ans)))
                    true_val = int(problem['answer'])
                    if pred_val == true_val:
                        correct += 1
                        vprint(f"✅ MATCH: Predicted {pred_val} == Truth {true_val} (Solving Time: {t_prob_spent:.1f}s)")
                    else:
                        vprint(f"❌ MISMATCH: Predicted {pred_val} != Truth {true_val} (Solving Time: {t_prob_spent:.1f}s)")
                except Exception as e:
                    vprint(f"❌ PARSE ERROR during eval: {ans} (Error: {e})")
            
            print(f"\n📊 FINAL TEST SCORE: {correct}/{total} ({(correct/total)*100:.1f}%)")
            
            if not MOCK_KAGGLE_TEST:
                # In Batch mode, we only run once and finish
                return 

            # In Interactive mode, ask to run again
            try:
                print("\n" + "="*40)
                choice = input("🔄 Run another test? (Enter to continue selecting size, or 'exit' to quit): ").strip().lower()
                if choice in ['exit', 'quit']:
                    break
                # We don't actually need to set anything here, 
                # because the next loop will trigger the input() inside test.py via reload
            except (EOFError, KeyboardInterrupt, IndexError):
                # Handle Jupyter kernel/deque errors gracefully
                break
            except Exception:
                break
            except EOFError:
                break
        return

    if KAGGLE_MODE and not MOCK_KAGGLE_TEST:
        # We can't actually do a true 2-pass over the Kaggle Iterator because it forces Sequential yields
        # But we CAN dynamically adjust time. 
        # Modifying for Kaggle Iterable: If Fast Pass fails, just run the Deep Grind immediately on that single iteration loop.
        def modified_aimo_predict(problem, _):
            ans = orchestrator.solve_dynamic(problem.text)
            return ans

        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(modified_aimo_predict)
        inference_server.serve()
        
    else:
        print("--- LOCAL TESTING MODE ---")
        # 1. Try to load from test.csv
        if os.path.exists("test.csv"):
            print("Found test.csv. Parsing local submission loop...")
            df = pl.read_csv("test.csv")
            answers = []
            for row in df.iter_rows(named=True):
                prob = row.get("problem", "What is 2 + 2?")
                print(f"Solving: {prob[:50]}...")
                ans = orchestrator.solve_dynamic(prob)
                answers.append({"id": row.get("id", len(answers)), "answer": ans})
            pl.DataFrame(answers).write_csv("submission.csv")
            print("submission.csv generated successfully.")
            
        # 2. Try to load from test.py (Dynamic Import)
        else:
            test_problems = []
            if os.path.exists("test.py"):
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("local_test", "test.py")
                    test_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(test_mod)
                    test_problems = getattr(test_mod, "TEST_PROBLEMS", [])
                except Exception as e:
                    print(f"Error loading test.py: {e}")
            
            if not test_problems:
                test_problems = [{"id": "BASIC", "problem": "What is 2 + 2?", "answer": 4}]
                print("No custom test.py with TEST_PROBLEMS found. Falling back to basic 2+2 check.")

            if test_problems:
                print(f"Running {len(test_problems)} custom test cases from test.py...")
                for case in test_problems:
                    print(f"\n[Case {case['id']}] Solving: {case['problem']}")
                    ans = orchestrator.solve_dynamic(case['problem'])
                    print(f"Result for {case['id']}: {ans} (Expected: {case.get('answer', 'N/A')})")

    # Phase 4: Clean Exit - Prevent Kaggle VRAM Leaks
    try:
        if 'orchestrator' in locals():
            orchestrator.kernel_pool.cleanup()
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

if __name__ == "__main__":
    main()
