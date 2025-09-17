#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.stats import multivariate_normal
import matplotlib.patches as patches
from scipy.spatial import Voronoi, voronoi_plot_2d

def create_sample_data():
    """Create sample true tweet points and predicted field for visualization"""
    np.random.seed(42)

    # Sample true tweet locations (clustered)
    cluster1 = np.random.normal([2, 3], [0.3, 0.4], (15, 2))
    cluster2 = np.random.normal([6, 1], [0.4, 0.2], (10, 2))
    cluster3 = np.random.normal([4, 5], [0.2, 0.3], (8, 2))
    true_points = np.vstack([cluster1, cluster2, cluster3])

    return true_points

def plot_gaussian_mixture_field():
    """Visualize Gaussian Mixture Field approach"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    true_points = create_sample_data()

    # Define Gaussian components (what model predicts)
    gaussians = [
        {'center': (2.1, 3.2), 'std_x': 0.5, 'std_y': 0.6, 'weight': 0.4, 'correlation': 0.1},
        {'center': (5.8, 1.1), 'std_x': 0.4, 'std_y': 0.3, 'weight': 0.35, 'correlation': -0.2},
        {'center': (4.2, 4.8), 'std_x': 0.3, 'std_y': 0.4, 'weight': 0.25, 'correlation': 0.0}
    ]

    # Create meshgrid for density visualization
    x = np.linspace(0, 8, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Plot 1: Individual Gaussians
    ax1.set_title('Individual Gaussian Components\n(What the model learns)', fontsize=14, fontweight='bold')

    for i, g in enumerate(gaussians):
        # Create covariance matrix
        cov = [[g['std_x']**2, g['correlation']*g['std_x']*g['std_y']],
               [g['correlation']*g['std_x']*g['std_y'], g['std_y']**2]]

        # Plot Gaussian contours
        rv = multivariate_normal(g['center'], cov)
        density = rv.pdf(pos)

        contours = ax1.contour(X, Y, density, levels=5, alpha=0.7,
                              colors=[plt.cm.Set1(i)], linewidths=2)
        ax1.clabel(contours, inline=True, fontsize=8)

        # Mark center
        ax1.plot(g['center'][0], g['center'][1], 'o', color=plt.cm.Set1(i),
                markersize=12, label=f'Component {i+1} (w={g["weight"]:.2f})')

        # Add ellipse showing 1-sigma boundary
        ellipse = Ellipse(g['center'], 2*g['std_x'], 2*g['std_y'],
                         angle=np.degrees(np.arctan2(2*g['correlation']*g['std_x']*g['std_y'],
                                                    g['std_x']**2 - g['std_y']**2))/2,
                         fill=False, color=plt.cm.Set1(i), linewidth=2, linestyle='--')
        ax1.add_patch(ellipse)

    ax1.scatter(true_points[:, 0], true_points[:, 1], c='black', s=20, alpha=0.7, label='True tweets')
    ax1.legend()
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Combined field
    ax2.set_title('Combined Probability Field\n(Final prediction)', fontsize=14, fontweight='bold')

    # Calculate combined density
    combined_density = np.zeros_like(X)
    for g in gaussians:
        cov = [[g['std_x']**2, g['correlation']*g['std_x']*g['std_y']],
               [g['correlation']*g['std_x']*g['std_y'], g['std_y']**2]]
        rv = multivariate_normal(g['center'], cov)
        combined_density += g['weight'] * rv.pdf(pos)

    # Heatmap of combined field
    im = ax2.imshow(combined_density, extent=[0, 8, 0, 6], origin='lower',
                   cmap='viridis', alpha=0.8)
    plt.colorbar(im, ax=ax2, label='Probability Density')

    # Overlay true points
    ax2.scatter(true_points[:, 0], true_points[:, 1], c='red', s=30,
               marker='x', linewidth=2, label='True tweets')
    ax2.legend()
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 6)

    # Plot 3: Evaluation visualization
    ax3.set_title('Field Evaluation\n(How well does it match?)', fontsize=14, fontweight='bold')

    # Show prediction confidence regions
    contour_levels = [0.02, 0.05, 0.1, 0.2]
    cs = ax3.contourf(X, Y, combined_density, levels=contour_levels,
                     cmap='YlOrRd', alpha=0.6)
    ax3.contour(X, Y, combined_density, levels=contour_levels, colors='black', linewidths=1)

    # Color-code true points by how well predicted
    for point in true_points:
        # Sample density at this point
        idx_x = int((point[0] / 8) * 99)
        idx_y = int((point[1] / 6) * 99)
        density_val = combined_density[idx_y, idx_x] if 0 <= idx_x < 100 and 0 <= idx_y < 100 else 0

        if density_val > 0.1:
            color, marker, label = 'green', 'o', 'Well predicted'
        elif density_val > 0.05:
            color, marker, label = 'orange', 's', 'Moderately predicted'
        else:
            color, marker, label = 'red', 'x', 'Poorly predicted'

        ax3.scatter(point[0], point[1], c=color, marker=marker, s=50,
                   edgecolor='black', linewidth=1)

    # Add legend for point colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Well predicted'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='Moderately predicted'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Poorly predicted', linestyle='None')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    plt.colorbar(cs, ax=ax3, label='Prediction Confidence')
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig('image_outputs/gaussian_mixture_field_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: gaussian_mixture_field_infographic.png")

def plot_kernel_density_field():
    """Visualize Kernel Density Field approach"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    true_points = create_sample_data()

    # Define prototype points (what model learns)
    prototypes = np.array([
        [2.0, 3.0], [2.2, 3.4], [2.1, 2.8],  # Cluster 1 prototypes
        [6.0, 1.0], [5.8, 1.2],              # Cluster 2 prototypes
        [4.1, 4.9], [4.3, 5.1]               # Cluster 3 prototypes
    ])
    weights = np.array([0.2, 0.15, 0.1, 0.25, 0.15, 0.1, 0.05])
    bandwidth = 0.4

    # Create meshgrid
    x = np.linspace(0, 8, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)

    # Plot 1: Prototype points and kernels
    ax1.set_title('Prototype Points & Kernels\n(What the model learns)', fontsize=14, fontweight='bold')

    for i, (prototype, weight) in enumerate(zip(prototypes, weights)):
        # Plot individual kernel
        kernel_density = weight * np.exp(-((X - prototype[0])**2 + (Y - prototype[1])**2) / (2 * bandwidth**2))

        if i < 3:  # Different colors for different clusters
            color = 'red'
            alpha = 0.3
        elif i < 5:
            color = 'blue'
            alpha = 0.3
        else:
            color = 'green'
            alpha = 0.3

        ax1.contour(X, Y, kernel_density, levels=3, colors=[color], alpha=alpha, linewidths=1)

        # Plot prototype point
        ax1.scatter(prototype[0], prototype[1], c=color, s=100*weight*10,
                   marker='o', edgecolor='black', linewidth=2, alpha=0.8,
                   label=f'Prototype {i+1} (w={weight:.2f})' if i < 3 else None)

        # Show kernel radius
        circle = Circle(prototype, bandwidth, fill=False, color=color,
                       linestyle='--', alpha=0.5, linewidth=1)
        ax1.add_patch(circle)

    ax1.scatter(true_points[:, 0], true_points[:, 1], c='black', s=20, alpha=0.7,
               marker='x', label='True tweets')
    ax1.legend()
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, 5.5, f'Bandwidth: {bandwidth}', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Plot 2: Combined kernel density field
    ax2.set_title('Combined Kernel Density Field\n(Final prediction)', fontsize=14, fontweight='bold')

    # Calculate combined density
    combined_density = np.zeros_like(X)
    for prototype, weight in zip(prototypes, weights):
        kernel_density = weight * np.exp(-((X - prototype[0])**2 + (Y - prototype[1])**2) / (2 * bandwidth**2))
        combined_density += kernel_density

    # Heatmap
    im = ax2.imshow(combined_density, extent=[0, 8, 0, 6], origin='lower',
                   cmap='plasma', alpha=0.8)
    plt.colorbar(im, ax=ax2, label='Kernel Density')

    # Overlay components
    ax2.scatter(prototypes[:, 0], prototypes[:, 1], c='white', s=50,
               marker='o', edgecolor='black', linewidth=1, label='Prototypes')
    ax2.scatter(true_points[:, 0], true_points[:, 1], c='lime', s=30,
               marker='x', linewidth=2, label='True tweets')
    ax2.legend()
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 6)

    # Plot 3: Distance-based evaluation
    ax3.set_title('Distance-Based Evaluation\n(Harshness adjustable)', fontsize=14, fontweight='bold')

    # Show distance circles around prototypes
    for prototype, weight in zip(prototypes, weights):
        # Multiple distance rings showing different harshness levels
        for radius, alpha, label in [(0.3, 0.7, 'Strict'), (0.6, 0.4, 'Balanced'), (1.0, 0.2, 'Lenient')]:
            circle = Circle(prototype, radius, fill=False, color='blue',
                           alpha=alpha, linewidth=2)
            ax3.add_patch(circle)

    # Color-code true points by distance to nearest prototype
    for point in true_points:
        distances = [np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2) for p in prototypes]
        min_distance = min(distances)

        if min_distance < 0.3:
            color, size, label = 'green', 60, 'Excellent (strict)'
        elif min_distance < 0.6:
            color, size, label = 'orange', 50, 'Good (balanced)'
        elif min_distance < 1.0:
            color, size, label = 'yellow', 40, 'Fair (lenient)'
        else:
            color, size, label = 'red', 30, 'Poor (all levels)'

        ax3.scatter(point[0], point[1], c=color, s=size, marker='o',
                   edgecolor='black', linewidth=1)

    # Add prototypes
    ax3.scatter(prototypes[:, 0], prototypes[:, 1], c='blue', s=80,
               marker='s', edgecolor='black', linewidth=2, label='Prototypes')

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Excellent (< 0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Good (< 0.6)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Fair (< 1.0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Poor (≥ 1.0)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Prototypes')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 6)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('image_outputs/kernel_density_field_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: kernel_density_field_infographic.png")

def plot_voronoi_field():
    """Visualize Voronoi/Watershed Field approach"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    true_points = create_sample_data()

    # Define region centers and properties
    regions = [
        {'center': (2.0, 3.0), 'radius': 1.2, 'density': 0.8},
        {'center': (6.0, 1.0), 'radius': 1.0, 'density': 0.6},
        {'center': (4.0, 5.0), 'radius': 0.8, 'density': 0.4},
        {'center': (1.0, 1.0), 'radius': 0.6, 'density': 0.2},  # Low density region
        {'center': (7.0, 4.0), 'radius': 0.7, 'density': 0.3}   # Medium density region
    ]

    # Extract centers for Voronoi diagram
    centers = np.array([r['center'] for r in regions])

    # Plot 1: Individual regions
    ax1.set_title('Individual Regions\n(What the model learns)', fontsize=14, fontweight='bold')

    for i, region in enumerate(regions):
        # Draw region boundary
        circle = Circle(region['center'], region['radius'],
                       fill=True, alpha=0.3, color=plt.cm.Set3(i),
                       edgecolor='black', linewidth=2)
        ax1.add_patch(circle)

        # Mark center
        ax1.scatter(region['center'][0], region['center'][1],
                   c='black', s=100, marker='o', edgecolor='white', linewidth=2)

        # Add density label
        ax1.text(region['center'][0], region['center'][1] - 0.2,
                f'ρ={region["density"]:.1f}', ha='center', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Add region label
        ax1.text(region['center'][0], region['center'][1] + region['radius'] + 0.2,
                f'Region {i+1}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=plt.cm.Set3(i), alpha=0.7))

    ax1.scatter(true_points[:, 0], true_points[:, 1], c='red', s=30,
               marker='x', linewidth=2, label='True tweets')
    ax1.legend()
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Voronoi tesselation
    ax2.set_title('Voronoi Tessellation\n(Spatial partitioning)', fontsize=14, fontweight='bold')

    # Create Voronoi diagram
    vor = Voronoi(centers)

    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax2, show_vertices=False, line_colors='black',
                    line_width=2, point_size=0)

    # Color regions by density
    for i, region in enumerate(regions):
        ax2.scatter(region['center'][0], region['center'][1],
                   c=region['density'], s=200, marker='o',
                   cmap='YlOrRd', vmin=0, vmax=1, edgecolor='black', linewidth=2)

        # Fill Voronoi cells (approximate with circles for visualization)
        circle = Circle(region['center'], region['radius']*0.8,
                       fill=True, alpha=region['density']*0.5, color='red')
        ax2.add_patch(circle)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Predicted Density')

    ax2.scatter(true_points[:, 0], true_points[:, 1], c='blue', s=25,
               marker='o', alpha=0.8, label='True tweets')
    ax2.legend()
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 6)

    # Plot 3: Region-based evaluation
    ax3.set_title('Region-Based Evaluation\n(Capture rate per region)', fontsize=14, fontweight='bold')

    # Calculate how many tweets each region captures
    region_captures = []
    region_scores = []

    for i, region in enumerate(regions):
        # Count tweets in this region
        distances = [np.sqrt((pt[0] - region['center'][0])**2 + (pt[1] - region['center'][1])**2)
                    for pt in true_points]
        captured_count = sum(1 for d in distances if d <= region['radius'])
        capture_rate = captured_count / len(true_points)

        # Calculate region score (capture rate * predicted density)
        score = capture_rate * region['density']
        region_captures.append(captured_count)
        region_scores.append(score)

        # Draw region with score-based coloring
        circle = Circle(region['center'], region['radius'],
                       fill=True, alpha=0.6, color=plt.cm.RdYlGn(score),
                       edgecolor='black', linewidth=2)
        ax3.add_patch(circle)

        # Add score text
        ax3.text(region['center'][0], region['center'][1],
                f'{captured_count}/{len(true_points)}\nScore: {score:.3f}',
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # Color-code true points by which region captured them
    for point in true_points:
        # Find which region (if any) captures this point
        captured = False
        for i, region in enumerate(regions):
            distance = np.sqrt((point[0] - region['center'][0])**2 + (point[1] - region['center'][1])**2)
            if distance <= region['radius']:
                ax3.scatter(point[0], point[1], c=plt.cm.Set3(i), s=40,
                           marker='o', edgecolor='black', linewidth=1)
                captured = True
                break

        if not captured:
            ax3.scatter(point[0], point[1], c='gray', s=40,
                       marker='x', linewidth=2)

    # Add performance summary
    total_score = sum(region_scores)
    capture_rate = sum(region_captures) / len(true_points)
    ax3.text(0.5, 5.5, f'Total Score: {total_score:.3f}\nCapture Rate: {capture_rate:.1%}',
            fontsize=12, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))

    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 6)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('image_outputs/voronoi_field_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: voronoi_field_infographic.png")

def plot_harshness_comparison():
    """Show how harshness parameter affects evaluation"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    true_points = create_sample_data()

    # Simple prediction: one Gaussian slightly off-center
    pred_center = (2.3, 3.5)  # Slightly offset from main cluster at (2, 3)

    harshness_levels = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    for idx, harshness in enumerate(harshness_levels):
        ax = axes[idx]

        # Create grid for visualization
        x = np.linspace(0, 8, 100)
        y = np.linspace(0, 6, 100)
        X, Y = np.meshgrid(x, y)

        # Calculate scores for each point based on distance to prediction
        distances = np.sqrt((X - pred_center[0])**2 + (Y - pred_center[1])**2)
        scores = np.exp(-harshness * distances)

        # Plot score field
        im = ax.imshow(scores, extent=[0, 8, 0, 6], origin='lower',
                      cmap='RdYlGn', vmin=0, vmax=1, alpha=0.7)

        # Plot prediction center
        ax.scatter(pred_center[0], pred_center[1], c='blue', s=200,
                  marker='*', edgecolor='black', linewidth=2, label='Prediction')

        # Color-code true points by their scores
        for point in true_points:
            distance = np.sqrt((point[0] - pred_center[0])**2 + (point[1] - pred_center[1])**2)
            score = np.exp(-harshness * distance)

            if score > 0.7:
                color, size = 'green', 60
            elif score > 0.3:
                color, size = 'orange', 50
            else:
                color, size = 'red', 40

            ax.scatter(point[0], point[1], c=color, s=size, marker='o',
                      edgecolor='black', linewidth=1, alpha=0.8)

        ax.set_title(f'Harshness = {harshness}\n'
                    f'{"Lenient" if harshness < 1 else "Strict" if harshness > 2 else "Balanced"}',
                    fontsize=12, fontweight='bold')
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.legend()

        # Calculate average score for this harshness level
        avg_score = np.mean([np.exp(-harshness * np.sqrt((pt[0] - pred_center[0])**2 + (pt[1] - pred_center[1])**2))
                            for pt in true_points])
        ax.text(0.5, 0.5, f'Avg Score: {avg_score:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # Add overall colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Score (0=poor, 1=perfect)')

    plt.tight_layout()
    plt.savefig('image_outputs/harshness_comparison_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: harshness_comparison_infographic.png")

if __name__ == "__main__":
    print("Creating field-based approach infographics...")

    plot_gaussian_mixture_field()
    plot_kernel_density_field()
    plot_voronoi_field()
    plot_harshness_comparison()

    print("\n🎯 All infographics created in image_outputs/:")
    print("   • gaussian_mixture_field_infographic.png")
    print("   • kernel_density_field_infographic.png")
    print("   • voronoi_field_infographic.png")
    print("   • harshness_comparison_infographic.png")

    print("\n📊 These show:")
    print("   1. How each field approach creates spatial predictions")
    print("   2. How predictions are evaluated against true tweet locations")
    print("   3. How harshness parameter controls evaluation strictness")