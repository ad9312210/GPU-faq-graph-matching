# -*- coding: utf-8 -*-
# ================================================================
#   DEBUG SCRIPT - Check what functions are in each module
#   and what signature they expect
# ================================================================

import sys
import os
import inspect
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_module(name, path):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(full_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def inspect_module(module_name, path):
    """Print all functions and their signatures."""
    print(f"\n{'=' * 70}")
    print(f"  MODULE: {module_name} ({path})")
    print("=" * 70)
    
    module = load_module(module_name, path)
    if module is None:
        print("  ❌ Failed to load")
        return
    
    # Find all callable functions
    functions = [name for name in dir(module)
                  if not name.startswith('_') and callable(getattr(module, name))]
    
    # Filter for likely main functions
    main_funcs = [f for f in functions
                   if any(kw in f.lower()
                          for kw in ['scgm', 'match', 'sparse', 'gpu', 'cpu'])]
    
    print(f"\n  Main functions ({len(main_funcs)}):")
    for fn_name in main_funcs:
        fn = getattr(module, fn_name)
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            print(f"\n  📌 {fn_name}({', '.join(params)})")
            
            # Show parameter details
            for pname, param in sig.parameters.items():
                default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
                print(f"     - {pname}{default}")
        except Exception as e:
            print(f"  ❌ {fn_name}: {e}")


def main():
    print("=" * 70)
    print("  MODULE SIGNATURE INSPECTOR")
    print("=" * 70)
    
    modules_to_check = [
        ("Sequential CPU", "01_cpu_sequential_sparse_matching.py"),
        ("Parallel CPU", "02_cpu_parallel_sparse_matching.py"),
        ("GPU", "03_gpu_sparse_matching.py"),
        ("GOAT", "goat_baseline.py"),
        ("True Sparse", "true_sparse_scgm.py"),
    ]
    
    for name, path in modules_to_check:
        inspect_module(name, path)
    
    print("\n" + "=" * 70)
    print("  DONE - Check output above for exact function signatures")
    print("=" * 70)


if __name__ == "__main__":
    main()