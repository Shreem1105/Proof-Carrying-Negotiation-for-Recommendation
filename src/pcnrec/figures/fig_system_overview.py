import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_figure(output_dir):
    """
    Generates Figure 3: System Overview Diagram.
    Uses matplotlib patches to draw a clean architectural diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')
    
    # Title Bar
    ax.add_patch(patches.Rectangle((0, 56), 100, 4, facecolor='#2c3e50', edgecolor='none', zorder=10))
    ax.text(50, 58, "PCN-Rec: Proof-Carrying Negotiation Pipeline", color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=11)
    
    # 1. Inputs (User + Model) - shifted right to reduce whitespace
    # Was x=5. Shift everything +5 or +10. Let's shift +5.
    
    # Component Helper
    def draw_box(x, y, w, h, text, color='#ecf0f1', edge='#bdc3c7', fontsize=10, bold=False):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=1.5, edgecolor=edge, facecolor=color, zorder=2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight=weight, zorder=3)
        return rect
        
    def draw_arrow(x1, y1, x2, y2, label=None, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=1.5, color='#7f8c8d'), zorder=1)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+1, label, ha='center', va='bottom', fontsize=8, color='#7f8c8d')
            
    # Draw Shield function
    def draw_shield(x, y, s=2):
        # Simple shield patch
        verts = [(x, y), (x+s, y), (x+s, y-s*1.2), (x+s/2, y-s*2), (x, y-s*1.2), (x, y)]
        poly = patches.Polygon(verts, facecolor='#55efc4', edgecolor='#00b894', zorder=4)
        ax.add_patch(poly)

    # 1. Input
    draw_box(10, 42, 12, 8, "User History\n& Candidate\nModel", color='#dfe6e9')
    draw_arrow(23, 46, 30, 46) # ->
    
    # 2. Window
    draw_box(30, 42, 10, 8, "Candidate\nWindow\n(W=80)", color='#ffeaa7', edge='#f1c40f')
    draw_arrow(41, 46, 45, 46) # ->
    
    # 3. Agents Area
    # Background
    ax.add_patch(patches.Rectangle((45, 23), 26, 30, linewidth=1, edgecolor='#95a5a6', facecolor='#f5f6fa', linestyle='--', zorder=0))
    ax.text(46, 51, "Negotiation Step", fontsize=9, color='#7f8c8d', style='italic')

    # Agents
    draw_box(47, 40, 8, 6, "User\nAdvocate", color='#a29bfe', edge='#6c5ce7', fontsize=9)
    draw_box(61, 40, 8, 6, "Policy\nAgent", color='#fab1a0', edge='#e17055', fontsize=9)
    
    draw_arrow(51, 40, 55, 32)
    draw_arrow(65, 40, 59, 32)
    
    # Mediator
    draw_box(53, 26, 10, 6, "Mediator\n(LLM)", color='#74b9ff', edge='#0984e3', bold=True)
    draw_arrow(64, 29, 73, 29, "Certificate")
    
    # 4. Verifier
    draw_box(73, 23, 12, 12, "Deterministic\nVerifier\n(Code)", color='#55efc4', edge='#00b894', bold=True)
    # Add Shield Icon
    draw_shield(73.5, 34, s=1.5) 
    
    # 5. Success Path
    draw_arrow(86, 29, 91, 29, "Pass")
    draw_box(91, 25, 9, 8, "Final\nRecommendations", color='#00b894', edge='#00b894', bold=True, fontsize=9)
    
    # 6. Fail / Repair Path
    draw_arrow(79, 23, 79, 16, "Fail")
    
    draw_box(71, 8, 16, 8, "Deterministic\nRepair\n(Constrained\nGreedy)", color='#ff7675', edge='#d63031')
    
    # Repair loops back to final
    # Complex arrow
    ax.annotate("", xy=(95, 24), xytext=(88, 12),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#d63031', connectionstyle="arc3,rad=0.2"), zorder=1)
    
    # 7. Legend / Callouts (Left side, improved)
    # Using the now empty space at x=0-10 effectively? Actually we shifted input to x=10.
    ax.text(2, 35, "Proof-Carrying:", fontsize=10, fontweight='bold', color='#2d3436')
    ax.text(2, 31, "Verifier rejects\nsafety violations\ninstantly.", fontsize=9, color='#636e72', verticalalignment='top')
    
    ax.text(2, 15, "Feasibility:", fontsize=10, fontweight='bold', color='#2d3436')
    ax.text(2, 11, "Window W controls\nsolvability vs.\nchoice size.", fontsize=9, color='#636e72', verticalalignment='top')

    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(output_dir, 'fig3_system_overview.png'))
    fig.savefig(os.path.join(output_dir, 'fig3_system_overview.pdf'))
    plt.close(fig)
    
    # Caption
    caption = (
        "Figure 3: PCN-Rec System Architecture. The system uses a negotiation between a User Advocate and "
        "Policy Agent, mediated by an LLM to produce a slate and a 'proof certificate'. "
        "A deterministic code-based verifier checks the certificate against constraints. "
        "If verification fails, a deterministic repair mechanism (Constrained Greedy) ensures compliance."
    )
    with open(os.path.join(output_dir, 'captions', 'fig3_caption.txt'), 'w') as f:
        f.write(caption)

    print(f"Generated Figure 3 in {output_dir}")
