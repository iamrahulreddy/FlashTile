"""
FlashTile Demo Application
==========================

Small Gradio UI for comparing memory use across the attention implementations in this repo.

Usage:
    python demo/app.py
    
Requirements:
    pip install gradio
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Gradio not installed. Install with: pip install gradio")
    raise SystemExit(1)

from flashtile import (
    NaiveAttention,
    FlashAttentionV1,
    FlashAttentionV2,
    GroupedQueryAttention,
    check_installation,
)


def _select_num_kv_heads(num_heads: int, target_ratio: int = 4) -> int:
    """Pick a valid GQA KV-head count that divides num_heads."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    target = max(1, num_heads // target_ratio)
    divisors = [d for d in range(1, num_heads + 1) if num_heads % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), -d))


def get_system_info():
    """Get system information."""
    info = check_installation()
    return f"""
    **System Information:**
    - PyTorch: {info['torch_version']}
    - CUDA Available: {info['cuda_available']}
    - CUDA Version: {info.get('cuda_version', 'N/A')}
    - Triton Available: {info['triton_available']}
    """


def run_comparison(
    embed_dim: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    implementation: str,
    causal: bool,
):
    """Run attention comparison and return results."""
    if not torch.cuda.is_available():
        return "CUDA not available. This demo works best with a GPU.", None, None
    
    if embed_dim % num_heads != 0:
        return (
            f"Invalid config: embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads}).",
            None,
            None,
        )

    device = "cuda"
    
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    results = []
    memory_data = {}
    output = None
    
    # Test requested implementation
    try:
        if implementation == "Flash V1":
            model = FlashAttentionV1(embed_dim, num_heads, causal=causal).to(device)
        elif implementation == "Flash V2":
            model = FlashAttentionV2(embed_dim, num_heads, causal=causal).to(device)
        elif implementation == "GQA":
            num_kv = _select_num_kv_heads(num_heads)
            model = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads=num_kv, causal=causal).to(device)
        else:  # Naive
            model = NaiveAttention(embed_dim, num_heads, causal=causal).to(device)
        
        model.eval()
        
        # Memory measurement
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            output, _ = model(x)
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Theoretical memory
        mem_info = model.get_memory_usage(batch_size, seq_len)
        
        results.append(f"**{implementation} Results:**")
        results.append(f"- Output shape: {output.shape}")
        results.append(f"- Peak memory: {memory_mb:.2f} MB")
        results.append(f"- Memory complexity: {mem_info.get('complexity', 'N/A')}")
        
        if 'comparison_with_naive' in mem_info:
            reduction = mem_info['comparison_with_naive']['reduction_ratio']
            results.append(f"- Reduction vs Naive: {reduction:.1f}x")
        
        memory_data[implementation] = memory_mb
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            results.append(f"**{implementation}:** Out of Memory")
            memory_data[implementation] = 0
        else:
            raise
    
    except ValueError as e:
        return f"Configuration error: {e}", None, None

    # Also test naive for comparison
    if implementation != "Naive":
        try:
            naive_model = NaiveAttention(embed_dim, num_heads, causal=causal).to(device)
            naive_model.eval()
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = naive_model(x)
            
            naive_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            memory_data["Naive (reference)"] = naive_memory_mb
            
            # Add comparison
            if memory_data.get(implementation, 0) > 0:
                reduction = naive_memory_mb / memory_data[implementation]
                results.append(f"\n**Comparison vs Naive:**")
                results.append(f"- Naive memory: {naive_memory_mb:.2f} MB")
                results.append(f"- **Reduction: {reduction:.1f}x**")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                memory_data["Naive (reference)"] = -1  # OOM marker
                results.append("\n**Naive:** Out of Memory (too large to compare)")
    
    output_np = output.cpu().numpy() if output is not None else None
    return "\n".join(results), memory_data, output_np


def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(title="FlashTile Demo") as demo:
        gr.Markdown("""
        # FlashTile: Flash Attention Demo
        
        Compare memory usage across the reference attention implementations in this repo.
        Use the controls below to see how sequence length, batch size, and implementation
        choice affect peak memory.
        """)
        
        with gr.Row():
            gr.Markdown(get_system_info())
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration")
                
                embed_dim = gr.Slider(
                    minimum=128,
                    maximum=1024,
                    value=512,
                    step=128,
                    label="Embedding Dimension",
                )
                
                num_heads = gr.Slider(
                    minimum=2,
                    maximum=16,
                    value=8,
                    step=2,
                    label="Number of Heads",
                )
                
                seq_len = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    value=1024,
                    step=256,
                    label="Sequence Length",
                )
                
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="Batch Size",
                )
                
                implementation = gr.Dropdown(
                    choices=["Naive", "Flash V1", "Flash V2", "GQA"],
                    value="Flash V2",
                    label="Implementation",
                )
                
                causal = gr.Checkbox(
                    value=True,
                    label="Causal Masking (for decoder models)",
                )
                
                run_btn = gr.Button("Run Comparison", variant="primary")
            
            with gr.Column():
                gr.Markdown("### Results")
                output_text = gr.Textbox(
                    label="Results",
                    lines=15,
                    interactive=False,
                )
                
                memory_plot = gr.BarPlot(
                    label="Memory Usage Comparison",
                    x="Implementation",
                    y="Memory (MB)",
                )
        
        with gr.Row():
            gr.Markdown("""
            ### About Flash Attention
            
            **Flash Attention** keeps memory at **O(N)** instead of **O(N^2)** by:
            
            1. **Tiling**: Processing attention in small blocks that fit in GPU SRAM
            2. **Online Softmax**: Computing softmax incrementally without materializing the full N x N matrix
            3. **Kernel Fusion**: Reducing memory traffic in optimized implementations
            
            This demo is mainly meant to compare the behavior of the implementations included in this repo.
            """)
        
        # Event handlers
        def on_run(embed_dim, num_heads, seq_len, batch_size, implementation, causal):
            text, memory_data, _ = run_comparison(
                embed_dim, num_heads, seq_len, batch_size, implementation, causal
            )
            
            # Format memory data for bar plot
            if memory_data:
                plot_data = {
                    "Implementation": list(memory_data.keys()),
                    "Memory (MB)": [max(0, v) for v in memory_data.values()],  # Handle -1 OOM marker
                }
            else:
                plot_data = None
            
            return text, plot_data
        
        run_btn.click(
            fn=on_run,
            inputs=[embed_dim, num_heads, seq_len, batch_size, implementation, causal],
            outputs=[output_text, memory_plot],
        )
    
    return demo


def main():
    """Main entry point."""
    print("=" * 60)
    print("FlashTile Interactive Demo")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available.\n")
        print("This demo works best with a GPU for meaningful memory comparisons.")
        print("CPU mode will work but won't show memory savings accurately.\n")
    
    demo = create_demo()
    
    print("\nStarting Gradio server...\n")
    print("Open your browser to the displayed URL\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()

