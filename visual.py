import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import traceback
import torch
from torch_geometric.data import DataLoader
from model import FHGNN
from dataset import HiMolGraph, MoleculeDataset


class HierarchicalMoleculeVisualizer:
    def __init__(self):
        self.red_cmap = LinearSegmentedColormap.from_list('red_gradient', 
                                                         [(1, 1, 1, 0), (1, 0.7, 0.7, 0.3), 
                                                          (1, 0.3, 0.3, 0.6), (1, 0, 0, 1)])
        self.blue_cmap = LinearSegmentedColormap.from_list('blue_gradient', 
                                                          [(1, 1, 1, 0), (0.7, 0.7, 1, 0.3), 
                                                           (0.3, 0.3, 1, 0.6), (0, 0, 1, 1)])
        self.green_cmap = LinearSegmentedColormap.from_list('green_gradient', 
                                                           [(1, 1, 1, 0), (0.7, 1, 0.7, 0.3), 
                                                            (0.3, 1, 0.3, 0.6), (0, 0.8, 0, 1)])
        
        self.atom_color_map = {'C': 'black', 'N': 'blue', 'O': 'red', 'S': 'goldenrod', 
                              'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'P': 'orange'}

    def extract_hierarchical_attention(self, args, model, batch, mol_idx=0, external_smiles=None):
        """
        Extracts hierarchical attention weights from a trained FHGNN model.

        Parameters:
            model (FHGNN): Pretrained FHGNN model instance
            batch (BatchMolGraph): Input batch of molecular graphs
            mol_idx (int): Index of target molecule in batch for visualization
            external_smiles (str, optional): External SMILES string for additional reference 

        Returns:
            attention_data: Dictionary containing hierarchical attention information
        """
        if external_smiles is not None:
            smiles = external_smiles
            print(f"Use of externally provided SMILES: {smiles}")
        else:
            try:
                smiles = batch.smi[mol_idx]
                
                if isinstance(smiles, (list, tuple)):
                    smiles = smiles[0] if len(smiles) > 0 else ""
                elif hasattr(smiles, 'item') and callable(getattr(smiles, 'item')):
                    smiles = smiles.item()
                elif hasattr(smiles, 'numpy') and callable(getattr(smiles, 'numpy')):
                    smiles_np = smiles.numpy()
                    if isinstance(smiles_np, np.ndarray):
                        smiles = ''.join([chr(x) for x in smiles_np if x != 0])
                    
                smiles = str(smiles)
                
                print(f"Parsed SMILES strings: {smiles}")
            except Exception as e:
                print(f"Error getting SMILES from batch: {e}")
                print("Try alternative approaches...")
                
                if hasattr(batch, 'smi') and hasattr(batch.smi, '__getitem__'):
                    try:
                        print(f"Type of batch.smi: {type(batch.smi)}")
                        
                        if torch.is_tensor(batch.smi):
                            if batch.smi.dim() == 2:
                                smiles_tensor = batch.smi[mol_idx]
                                smiles = ''.join([chr(x) for x in smiles_tensor.cpu().numpy() if x != 0])
                            elif batch.smi.dim() == 1:
                                smiles = str(batch.smi[mol_idx].item())
                        else:
                            dataset_path = os.path.join('/', args.dataset, 'processed/smiles.csv')
                            if os.path.exists(dataset_path):
                                import pandas as pd
                                smiles_list = pd.read_csv(dataset_path, header=None)[0].tolist()
                                smiles = smiles_list[mol_idx]
                                print(f"SMILES from smiles.csv: {smiles}")
                            else:
                                raise ValueError("Unable to get a valid SMILES string")
                    except Exception as inner_e:
                        print(f"Alternative way to get SMILES fails: {inner_e}")
                        raise ValueError("Unable to get a valid SMILES string")
        
        mol_batch = model.get_batch_mol_graph(batch)
        
        _, attention_patterns = model.encoder.forward(mol_batch, return_attention=True)
        
        attention_weights = model.encoder.cross_attention.attention_weights
        mol_key = f'molecule_{mol_idx}'
        
        if mol_key not in attention_weights:
            raise ValueError(f"Can't find the attention weight for the numerator {mol_idx}")
        
        mol_attention = attention_weights[mol_key]
        weights = mol_attention['weights']
        node_types = mol_attention['node_types']
        num_atoms = mol_attention['num_atoms']
        num_motifs = mol_attention['num_motifs']
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES: {smiles}")
            
        mol_graph = HiMolGraph(mol)
        cliques = mol_graph.cliques if hasattr(mol_graph, 'cliques') else []
        
        attention_data = {
            'smiles': smiles,
            'mol': mol,
            'weights': weights,
            'node_types': node_types,
            'num_atoms': num_atoms,
            'num_motifs': num_motifs,
            'cliques': cliques,
            'atom_attention': weights[:num_atoms, :num_atoms],
            'atom_motif_attention': weights[:num_atoms, num_atoms:num_atoms+num_motifs] if num_motifs > 0 else None,
            'motif_motif_attention': weights[num_atoms:num_atoms+num_motifs, num_atoms:num_atoms+num_motifs] if num_motifs > 0 else None,
            'atom_global_attention': weights[:num_atoms, -1],
            'motif_global_attention': weights[num_atoms:num_atoms+num_motifs, -1] if num_motifs > 0 else None
        }
        
        return attention_data
    
    def visualize_hierarchical_attention(self, attention_data, property_name="性质", save_path=None, figsize=(15, 18)):
        """
        Visualizes attention for hierarchical molecular graphs

        Parameters:
            attention_data: Attention data obtained from extract_hierarchical_attention
            property_name: Name of the property to display
            save_path: Path to save the image
            figsize: Figure size
        """
        smiles = attention_data['smiles']
        mol = attention_data['mol']
        num_atoms = attention_data['num_atoms']
        num_motifs = attention_data['num_motifs']
        cliques = attention_data['cliques']
        atom_global_attention = attention_data['atom_global_attention']
        motif_global_attention = attention_data['motif_global_attention']
        
        fig = plt.figure(figsize=figsize)
        
        if num_motifs > 0:
            ax_atom = plt.subplot2grid((3, 1), (0, 0))
            ax_motif = plt.subplot2grid((3, 1), (1, 0))
            ax_integrated = plt.subplot2grid((3, 1), (2, 0))
        else:
            ax_atom = plt.subplot2grid((2, 1), (0, 0))
            ax_integrated = plt.subplot2grid((2, 1), (1, 0))
            ax_motif = None

        
        self._visualize_atom_attention(ax_atom, mol, atom_global_attention, property_name)
        
        if num_motifs > 0 and ax_motif is not None:
            self._visualize_motif_attention(ax_motif, mol, cliques, motif_global_attention, property_name)
        
        self._visualize_integrated_view(ax_integrated, attention_data)
        ax_integrated.set_title("Hierarchical molecular graph attention integration view", fontsize=14)
        
        plt.suptitle(f"Hierarchical Attention Analysis of Molecules {smiles} - {property_name}", fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return fig
    
    def _visualize_atom_attention(self, ax, mol, atom_attention_scores, property_name):
        """Visualizing the Attention of the Atomic Layer"""
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)
        mol = Chem.RemoveHs(mol)
        
        conf = mol.GetConformer()
        atom_coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())])
        
        x_min, y_min = np.min(atom_coords, axis=0) - 1.5
        x_max, y_max = np.max(atom_coords, axis=0) + 1.5
        
        grid_size = 300
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((grid_size, grid_size))
        
        min_score = atom_attention_scores.min()
        max_score = atom_attention_scores.max()

        normalized_scores = (atom_attention_scores - min_score) / (max_score - min_score + 1e-6)

        normalized_scores = 2 * normalized_scores - 1

        
        for i, (atom_x, atom_y) in enumerate(atom_coords):
            if i >= len(normalized_scores):
                break
                
            distance = np.sqrt((X - atom_x)**2 + (Y - atom_y)**2)
            
            sigma = 0.3 * (np.abs(normalized_scores[i])**0.5 + 0.1)
            attention = normalized_scores[i] * np.exp(-distance**2 / (2 * sigma**2))
            
            Z += attention
        
        Z = np.clip(Z, -1, 1)
        
        pos_Z = np.copy(Z)
        pos_Z[pos_Z < 0] = 0
        if np.max(pos_Z) > 0:
            levels = np.linspace(0.05, np.max(pos_Z), 20)
            ax.contourf(X, Y, pos_Z, levels=levels, cmap=self.red_cmap, alpha=0.9)
            
            contour_levels = np.linspace(0.1, np.max(pos_Z), 10)
            ax.contour(X, Y, pos_Z, levels=contour_levels, colors='red', alpha=0.3, linewidths=0.5)
        
        neg_Z = np.copy(Z)
        neg_Z[neg_Z > 0] = 0
        neg_Z = -neg_Z
        if np.max(neg_Z) > 0:
            levels = np.linspace(0.05, np.max(neg_Z), 20)
            ax.contourf(X, Y, neg_Z, levels=levels, cmap=self.blue_cmap, alpha=0.9)
            
            contour_levels = np.linspace(0.1, np.max(neg_Z), 10)
            ax.contour(X, Y, neg_Z, levels=contour_levels, colors='blue', alpha=0.3, linewidths=0.5)
        
        self._draw_molecule_on_axis(ax, mol, atom_coords)
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')

    def _visualize_motif_attention(self, ax, mol, cliques, motif_attention_scores, property_name):
        """
        Motif Attention Visualization - Adds base colors to atoms and bonds
        The intensity of base colors directly represents motif attention scores, with red indicating positive contributions and blue indicating negative contributions
        Darker colors represent higher attention scores
        """
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        
        atom_coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())])
        x_min, y_min = np.min(atom_coords, axis=0) - 1.5
        x_max, y_max = np.max(atom_coords, axis=0) + 1.5
        
        if len(motif_attention_scores) == 0:
            print("Warning: no base-order attention scores")
            self._draw_molecule_on_axis(ax, mol, atom_coords, bond_width=1.0)
            ax.set_aspect('equal')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axis('off')
            return
        
        print(f"Range of motif attention scores: {np.min(motif_attention_scores):.3f} 到 {np.max(motif_attention_scores):.3f}")
        
        mean_score = np.mean(motif_attention_scores)
        print(f"Mean value of base-sequence attention scores: {mean_score:.3f}")
        
        normalized_scores = np.zeros_like(motif_attention_scores)
        
        if np.max(motif_attention_scores) == np.min(motif_attention_scores):
            print("Warning: all base-order attention scores are equal")
            if np.mean(motif_attention_scores) > 0.5:
                normalized_scores = np.ones_like(motif_attention_scores) * 0.5
            else:
                normalized_scores = np.ones_like(motif_attention_scores) * -0.5
        else:
            for i in range(len(motif_attention_scores)):
                if motif_attention_scores[i] > mean_score:
                    normalized_scores[i] = (motif_attention_scores[i] - mean_score) / (np.max(motif_attention_scores) - mean_score + 1e-6)
                else:
                    normalized_scores[i] = -(mean_score - motif_attention_scores[i]) / (mean_score - np.min(motif_attention_scores) + 1e-6)
        
        print(f"Normalized base-order attention score: {np.min(normalized_scores):.3f} to {np.max(normalized_scores):.3f}")
        
        atom_to_motif = {}
        motif_info = {}
        
        for motif_idx, motif in enumerate(cliques):
            if motif_idx >= len(normalized_scores):
                break
                
            score = normalized_scores[motif_idx]
            
            intensity = min(abs(score), 1.0)
            
            if score > 0:
                r = 1.0
                g = 1.0 - intensity
                b = 1.0 - intensity
                color = (r, g, b)
            else:
                r = 1.0 - intensity
                g = 1.0 - intensity
                b = 1.0
                color = (r, g, b)
            
            motif_info[motif_idx] = {
                'color': color,
                'atoms': motif,
                'score': score
            }
            
            for atom_idx in motif:
                atom_to_motif[atom_idx] = motif_idx
        
        self._draw_bond_backgrounds(ax, mol, atom_coords, motif_info, atom_to_motif, zorder=1)
        
        self._draw_atom_backgrounds(ax, mol, atom_coords, motif_info, atom_to_motif, zorder=2)
        
        self._draw_molecule_on_axis(ax, mol, atom_coords, bond_width=1.0, zorder=10)
        
        for motif_idx, motif in enumerate(cliques):
            if motif_idx >= len(motif_attention_scores):
                break
                
            motif_coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in motif])
            center = np.mean(motif_coords, axis=0)
            
            orig_score = motif_attention_scores[motif_idx]
            text = ax.text(center[0], center[1], f"M{motif_idx}\n{orig_score:.2f}", 
                ha='center', va='center', fontsize=9, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0, edgecolor=None, boxstyle='round,pad=0.5', linewidth=0),
                zorder=30)
            
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')

    def _draw_bond_backgrounds(self, ax, mol, atom_coords, motif_info, atom_to_motif, zorder=1):
        """
        Adding Undertones to Bonds in Molecules
        """
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            if begin_idx not in atom_to_motif or end_idx not in atom_to_motif:
                continue
            if atom_to_motif[begin_idx] != atom_to_motif[end_idx]:
                continue
                
            motif_idx = atom_to_motif[begin_idx]
            motif_data = motif_info[motif_idx]
            
            color = motif_data['color']
            
            begin_pos = atom_coords[begin_idx]
            end_pos = atom_coords[end_idx]
            
            dx = end_pos[0] - begin_pos[0]
            dy = end_pos[1] - begin_pos[1]
            bond_length = np.sqrt(dx**2 + dy**2)
            norm_dx, norm_dy = dx / bond_length, dy / bond_length
            
            perp_dx, perp_dy = -norm_dy, norm_dx
            
            width = 0.25
            
            p1 = begin_pos + np.array([perp_dx, perp_dy]) * width
            p2 = begin_pos - np.array([perp_dx, perp_dy]) * width
            p3 = end_pos - np.array([perp_dx, perp_dy]) * width
            p4 = end_pos + np.array([perp_dx, perp_dy]) * width
            
            poly = plt.Polygon([p1, p2, p3, p4], facecolor=color, edgecolor='none', 
                            alpha=1.0, zorder=zorder)
            ax.add_patch(poly)

    def _draw_atom_backgrounds(self, ax, mol, atom_coords, motif_info, atom_to_motif, zorder=2):
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx not in atom_to_motif:
                continue
                
            motif_idx = atom_to_motif[atom_idx]
            motif_data = motif_info[motif_idx]
            
            color = motif_data['color']
            
            atom_pos = atom_coords[atom_idx]
            
            background = plt.Circle(atom_pos, 0.4, facecolor=color, edgecolor='none', 
                                alpha=1.0, zorder=zorder)
            ax.add_patch(background)


    def _visualize_integrated_view(self, ax, attention_data):
        """
        Visualizes integrated view showing hierarchical structure and attention interactions between nodes

        Implementation:
            1. Fully-connected attention edges between atoms-motifs
            2. Fully-connected attention edges between motifs-motifs 
            3. Uses red/blue colors for positive/negative attention
            4. Edge darkness/thickness scales with attention score magnitude
        """
        import matplotlib.lines as mlines
        import matplotlib.patheffects as path_effects
        
        mol = attention_data['mol']
        num_atoms = attention_data['num_atoms']
        num_motifs = attention_data['num_motifs']
        cliques = attention_data['cliques']
        weights = attention_data['weights']
        
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        
        atom_coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in range(mol.GetNumAtoms())])
        atom_coords_scaled = atom_coords.copy()
        atom_coords_scaled[:, 1] = atom_coords_scaled[:, 1] - 4
        
        motif_centers = []
        for motif in cliques:
            if len(motif) > 0:
                motif_coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in motif])
                center = np.mean(motif_coords, axis=0)
                center[1] += 2
                motif_centers.append(center)
        
        mol_center = np.mean(atom_coords, axis=0)
        mol_center[1] += 6
        
        x_min, y_min = np.min(atom_coords_scaled, axis=0) - 2
        x_max, y_max = np.max(atom_coords_scaled, axis=0) + 2
        y_max = max(y_max, mol_center[1] + 2)
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        
        if num_motifs > 0 and len(motif_centers) > 0:
            min_atom_motif_weight = float('inf')
            max_atom_motif_weight = float('-inf')
            
            atom_motif_weights = []
            for atom_idx in range(num_atoms):
                for motif_idx in range(num_motifs):
                    if atom_idx < weights.shape[0] and num_atoms + motif_idx < weights.shape[1]:
                        weight = weights[atom_idx, num_atoms + motif_idx]
                        atom_motif_weights.append(weight)

            if atom_motif_weights:
                min_atom_motif_weight = min(atom_motif_weights)
                max_atom_motif_weight = max(atom_motif_weights)
                if max_atom_motif_weight == min_atom_motif_weight:
                    max_atom_motif_weight = min_atom_motif_weight + 1e-6

            for atom_idx in range(num_atoms):
                for motif_idx in range(num_motifs):
                    if motif_idx < len(motif_centers) and atom_idx < weights.shape[0] and num_atoms + motif_idx < weights.shape[1]:
                        weight = weights[atom_idx, num_atoms + motif_idx]

                        if abs(weight) > 0.03:
                            norm_weight = (weight - min_atom_motif_weight) / (max_atom_motif_weight - min_atom_motif_weight)
                            norm_weight = norm_weight * 2 - 1
                            norm_weight = max(-1, min(1, norm_weight))

                            alpha = 0.1 + 0.3 * abs(norm_weight)
                            linewidth = 0.1 + 1.5 * abs(norm_weight)
                            color = 'red' if norm_weight > 0 else 'blue'

                            line = mlines.Line2D(
                                [atom_coords_scaled[atom_idx, 0], motif_centers[motif_idx][0]],
                                [atom_coords_scaled[atom_idx, 1], motif_centers[motif_idx][1]],
                                color=color, alpha=alpha, linewidth=linewidth, zorder=1, linestyle=":"
                            )
                            ax.add_line(line)

            
            if num_motifs > 1:
                motif_motif_weights = []
                for i in range(num_motifs):
                    for j in range(num_motifs):
                        if i != j and num_atoms + i < weights.shape[0] and num_atoms + j < weights.shape[1]:
                            weight = weights[num_atoms + i, num_atoms + j]
                            motif_motif_weights.append(weight)
                
                min_motif_motif_weight = min(motif_motif_weights) if motif_motif_weights else 0
                max_motif_motif_weight = max(motif_motif_weights) if motif_motif_weights else 0.1
                
                if max_motif_motif_weight == min_motif_motif_weight:
                    max_motif_motif_weight = min_motif_motif_weight + 0.1
                
                for i in range(num_motifs):
                    for j in range(i+1, num_motifs):
                        if i < len(motif_centers) and j < len(motif_centers) and num_atoms + i < weights.shape[0] and num_atoms + j < weights.shape[1]:
                            weight_ij = weights[num_atoms + i, num_atoms + j]
                            weight_ji = weights[num_atoms + j, num_atoms + i]
                            weight = (weight_ij + weight_ji) / 2
                            
                            if abs(weight) > 0.01:
                                if min_motif_motif_weight == max_motif_motif_weight:
                                    norm_weight = 0.5
                                else:
                                    norm_weight = (abs(weight) - min_motif_motif_weight) / (max_motif_motif_weight - min_motif_motif_weight)
                                    norm_weight = max(0, min(1, norm_weight))
                                    norm_weight = 2 * norm_weight - 1

                                alpha = 0.1 + 0.8 * abs(norm_weight)
                                color = 'red' if norm_weight > 0 else 'blue'
                                linewidth = 1 + 2.0 * abs(norm_weight)
                                
                                line = mlines.Line2D(
                                    [motif_centers[i][0], motif_centers[j][0]],
                                    [motif_centers[i][1], motif_centers[j][1]],
                                    color=color, alpha=alpha, linewidth=linewidth, zorder=2,
                                    linestyle='--'
                                )
                                ax.add_line(line)
                                            
                motif_to_graph_weights = [
                    weights[num_atoms + motif_idx, -1]
                    for motif_idx in range(num_motifs)
                    if num_atoms + motif_idx < weights.shape[0]
                ]
                min_m2g = min(motif_to_graph_weights)
                max_m2g = max(motif_to_graph_weights)

                if min_m2g == max_m2g:
                    max_m2g += 1e-6

                for motif_idx, center in enumerate(motif_centers):
                    if motif_idx < num_motifs and num_atoms + motif_idx < weights.shape[0] and weights.shape[1] > 0:
                        weight = weights[num_atoms + motif_idx, -1]

                        norm_weight = (weight - min_m2g) / (max_m2g - min_m2g)
                        norm_weight = norm_weight * 2 - 1
                        norm_weight = max(-1, min(1, norm_weight))

                        alpha = 0.1 + 0.8 * abs(norm_weight)
                        linewidth = 0.5 + 2.0 * abs(norm_weight)
                        color = 'red' if norm_weight > 0 else 'blue'

                        line = mlines.Line2D(
                            [center[0], mol_center[0]],
                            [center[1], mol_center[1]],
                            color=color, alpha=alpha, linewidth=linewidth, zorder=2
                        )
                        ax.add_line(line)

        
            if num_motifs == 0:
                for atom_idx, coord in enumerate(atom_coords_scaled):
                    if atom_idx < num_atoms and atom_idx < weights.shape[0] and weights.shape[1] > 0:
                        weight = weights[atom_idx, -1] if -1 < weights.shape[1] else 0
                        
                        norm_weight = min(1.0, max(0.0, abs(weight)))
                        alpha = 0.1 + 0.8 * norm_weight
                        color = 'red' if weight > 0 else 'blue'
                        linewidth = 0.5 + 1.0 * norm_weight
                        
                        line = mlines.Line2D(
                            [coord[0], mol_center[0]],
                            [coord[1], mol_center[1]],
                            color=color, alpha=alpha, linewidth=linewidth, zorder=2
                        )
                        ax.add_line(line)
        
        self._draw_molecule_on_axis(ax, mol, atom_coords_scaled, bond_width=1.0, zorder=5)
                
        for motif_idx, center in enumerate(motif_centers):
            if motif_idx < num_motifs:
                circle = plt.Circle(center, 0.7, alpha=0.9, facecolor='lightblue', 
                                edgecolor='black', linewidth=1, zorder=10)
                ax.add_patch(circle)
                text = ax.text(center[0], center[1], f"M{motif_idx}", ha='center', va='center', 
                            fontsize=10, fontweight='bold', zorder=12)
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
        
        circle = plt.Circle(mol_center, 1.0, alpha=0.9, facecolor='orange', 
                        edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        text = ax.text(mol_center[0], mol_center[1], "G", ha='center', va='center', 
                    fontsize=12, fontweight='bold', zorder=12)
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    def _draw_colored_molecule(self, ax, mol, atom_coords, atom_colors, atom_to_motif, bond_width=1.5, zorder=10):
        """
        Draws molecular structure colored by attention scores

        Parameters:
            ax: matplotlib axis object
            mol: RDKit molecule object
            atom_coords: Array of atom coordinates
            atom_colors: Atom color mapping {atom_idx: color}
            atom_to_motif: Atom to motif mapping {atom_idx: motif_idx}
            bond_width: Bond width
            zorder: Drawing layer order
        """
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_pos = atom_coords[begin_idx]
            end_pos = atom_coords[end_idx]
            
            if begin_idx in atom_to_motif and end_idx in atom_to_motif and atom_to_motif[begin_idx] == atom_to_motif[end_idx]:
                bond_color = atom_colors[begin_idx]
            else:
                bond_color = 'black'
            
            dx = end_pos[0] - begin_pos[0]
            dy = end_pos[1] - begin_pos[1]
            bond_length = np.sqrt(dx**2 + dy**2)
            perpendicular = np.array([-dy, dx]) / bond_length * 0.15
            
            if bond.GetBondType() == Chem.BondType.SINGLE:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                        '-', color=bond_color, linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                        '-', color=bond_color, linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
                
                offset = perpendicular
                begin_pos2 = begin_pos + offset
                end_pos2 = end_pos + offset
                line = ax.plot([begin_pos2[0], end_pos2[0]], [begin_pos2[1], end_pos2[1]], 
                        '-', color=bond_color, linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                for offset_factor in [-1, 0, 1]:
                    offset = perpendicular * offset_factor
                    begin_pos_offset = begin_pos + offset
                    end_pos_offset = end_pos + offset
                    line = ax.plot([begin_pos_offset[0], end_pos_offset[0]], 
                            [begin_pos_offset[1], end_pos_offset[1]], 
                            '-', color=bond_color, linewidth=bond_width, zorder=zorder)[0]
                    line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.AROMATIC:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                        '-', color=bond_color, linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
                
                offset = perpendicular * 0.7
                begin_pos_offset = begin_pos + offset
                end_pos_offset = end_pos + offset
                line = ax.plot([begin_pos_offset[0], end_pos_offset[0]], 
                        [begin_pos_offset[1], end_pos_offset[1]], 
                        '--', color=bond_color, linewidth=bond_width-0.5, dashes=(3, 3), zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width, foreground='white')])
        
        for i, (x, y) in enumerate(atom_coords):
            if i < mol.GetNumAtoms():
                atom = mol.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                
                if i in atom_colors:
                    atom_color = atom_colors[i]
                else:
                    atom_color = self.atom_color_map.get(symbol, 'black')
                
                text = ax.text(x, y, symbol, fontsize=12, ha='center', va='center', 
                        color=atom_color, fontweight='bold', zorder=zorder+1)
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    def _draw_molecule_on_axis(self, ax, mol, atom_coords, bond_width=1.5, zorder=10):
        """Plot molecular structure on specified axis, add zorder parameter to control hierarchy"""
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_pos = atom_coords[begin_idx]
            end_pos = atom_coords[end_idx]
            
            dx = end_pos[0] - begin_pos[0]
            dy = end_pos[1] - begin_pos[1]
            bond_length = np.sqrt(dx**2 + dy**2)
            perpendicular = np.array([-dy, dx]) / bond_length * 0.15
            
            if bond.GetBondType() == Chem.BondType.SINGLE:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                            'k-', linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                            'k-', linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
                
                offset = perpendicular
                begin_pos2 = begin_pos + offset
                end_pos2 = end_pos + offset
                line = ax.plot([begin_pos2[0], end_pos2[0]], [begin_pos2[1], end_pos2[1]], 
                            'k-', linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                for offset_factor in [-1, 0, 1]:
                    offset = perpendicular * offset_factor
                    begin_pos_offset = begin_pos + offset
                    end_pos_offset = end_pos + offset
                    line = ax.plot([begin_pos_offset[0], end_pos_offset[0]], 
                                [begin_pos_offset[1], end_pos_offset[1]], 
                                'k-', linewidth=bond_width, zorder=zorder)[0]
                    line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
            
            elif bond.GetBondType() == Chem.BondType.AROMATIC:
                line = ax.plot([begin_pos[0], end_pos[0]], [begin_pos[1], end_pos[1]], 
                            'k-', linewidth=bond_width, zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width+1, foreground='white')])
                
                offset = perpendicular * 0.7
                begin_pos_offset = begin_pos + offset
                end_pos_offset = end_pos + offset
                line = ax.plot([begin_pos_offset[0], end_pos_offset[0]], 
                            [begin_pos_offset[1], end_pos_offset[1]], 
                            'k--', linewidth=bond_width-0.5, dashes=(3, 3), zorder=zorder)[0]
                line.set_path_effects([path_effects.withStroke(linewidth=bond_width, foreground='white')])
        
        for i, (x, y) in enumerate(atom_coords):
            if i < mol.GetNumAtoms():
                atom = mol.GetAtomWithIdx(i)
                symbol = atom.GetSymbol()
                color = self.atom_color_map.get(symbol, 'black')
                
                text = ax.text(x, y, symbol, fontsize=12, ha='center', va='center', 
                            color=color, fontweight='bold', zorder=zorder+1)
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    def _draw_attention_arrow(self, ax, start, end, weight, arrow_width=0.05, head_width=0.15,
                             color_pos='red', color_neg='blue'):
        """Drawing arrows with attention weights"""
        if weight > 0.5:
            color = color_pos
            width = arrow_width * (weight - 0.5) * 4  # 0.5->0, 1->2*arrow_width
        else:
            color = color_neg
            width = arrow_width * (0.5 - weight) * 4  # 0.5->0, 0->2*arrow_width
        
        if abs(weight - 0.5) < 0.05:
            return
        
        arrow = FancyArrowPatch(
            start, end, 
            arrowstyle=f'simple,head_width={head_width},head_length={head_width*1.5}',
            linewidth=width, 
            color=color, 
            alpha=min(1.0, abs(weight - 0.5) * 4),
            zorder=5
        )
        ax.add_patch(arrow)


def visual():
    parser = argparse.ArgumentParser(description='Visualizing the attention of hierarchical molecular maps')
    parser.add_argument('--device', type=int, default=0, help='GPU devices used')
    parser.add_argument('--model_path', type=str, required=True, help='Model Weights File Path')
    parser.add_argument('--dataset', type=str, default='bbbp', help='Data sets used')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Dataset Catalog')
    parser.add_argument('--all_mols', action='store_true', help='Whether to visualize all molecules')
    parser.add_argument('--max_mols', type=int, default=100, help='Maximum number of processed molecules')
    parser.add_argument('--mol_idx', type=int, default=0, help='Molecule index to visualize (used when --all_mols is False)')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--save_dir', type=str, default='./attention_viz/', help='Visualization results save directory')
    parser.add_argument('--property_name', type=str, default='molecular biology', help='The name of the property to be displayed')
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Use the device: {device}")

    os.makedirs(os.path.join(args.save_dir,args.dataset), exist_ok=True)
    
    print(f"Load Dataset: {args.dataset}")
    dataset = MoleculeDataset(os.path.join(args.data_dir, args.dataset), dataset=args.dataset)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")
    
    print("Create Model...")
    model = FHGNN(data_name=args.dataset, atom_fdim=89, bond_fdim=98, fp_fdim=6338, 
                 hidden_size=512, depth=7, device=device, out_dim=num_tasks)
    
    print(f"Loading model weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    visualizer = HierarchicalMoleculeVisualizer()
    
    smiles_path = os.path.join(args.data_dir, args.dataset, 'processed/smiles.csv')
    all_smiles = None
    if os.path.exists(smiles_path):
        import pandas as pd
        all_smiles = pd.read_csv(smiles_path, header=None)[0].tolist()
        all_smiles_len = len(all_smiles)
        print(f"Loaded {all_smiles_len} string of SMILES from CSV")
    
    total_mols = len(dataset)
    processed_mols = min(total_mols, args.max_mols) if args.all_mols else 1
    print(f"The dataset contains {total_mols} molecules, which will be processed {'all' if args.all_mols else 'specified'} molecules")
    
    if args.all_mols:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc=None):
                if desc:
                    print(desc)
                return iterable
        
        mol_count = 0
        
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batch")):
            batch = batch.to(device)
            for i in range(batch.id.shape[0]):
                if mol_count >= args.max_mols:
                    print(f"Reach maximum number of processed molecules: {args.max_mols}")
                    break
                    
                try:
                    global_mol_idx = batch_idx * args.batch_size + i
                    
                    external_smiles = all_smiles[global_mol_idx] if all_smiles and global_mol_idx < len(all_smiles) else None
                    
                    attention_data = visualizer.extract_hierarchical_attention(
                        args, model, batch, i, external_smiles=external_smiles
                    )
                    
                    save_path = os.path.join(args.save_dir, args.dataset, f"{args.dataset}_mol_{global_mol_idx}.png")
                    
                    visualizer.visualize_hierarchical_attention(
                        attention_data,
                        property_name=args.property_name,
                        save_path=save_path
                    )
                    
                    print(f"Generate an attention visualization for the molecule {global_mol_idx} and save it to: {save_path}")
                    mol_count += 1
                    
                except Exception as e:
                    print(f"Error processing molecule {global_mol_idx}: {e}")
                    traceback.print_exc()
                    continue
                    
            if mol_count >= args.max_mols:
                break
                
        print(f"Total {mol_count} molecules visualized")
    
    else:
        mol_idx = args.mol_idx
        print(f"Getting attention data for molecule {mol_idx}...")
        
        for i, batch in enumerate(loader):
            batch_size = batch.id.shape[0]
            if mol_idx < batch_size:
                batch = batch.to(device)
                
                try:
                    external_smiles = all_smiles[args.mol_idx] if all_smiles and args.mol_idx < len(all_smiles) else None
                    
                    attention_data = visualizer.extract_hierarchical_attention(
                        args, model, batch, mol_idx % batch_size,
                        external_smiles=external_smiles
                    )
                    
                    save_path = os.path.join(args.save_dir, f"{args.dataset}_mol_{args.mol_idx}.png")
                    print(f"Generate an attention visualization for the molecule {args.mol_idx}...")
                    
                    visualizer.visualize_hierarchical_attention(
                        attention_data,
                        property_name=args.property_name,
                        save_path=save_path
                    )
                    
                    print(f"The attention visualization has been saved to: {save_path}")
                    break
                    
                except Exception as e:
                    print(f"Errors occur during visualization: {e}")
                    traceback.print_exc()
                    break
                    
            mol_idx -= batch_size
        else:
            print(f"Molecule with index {args.mol_idx} not found.")


if __name__ == "__main__":
    visual()