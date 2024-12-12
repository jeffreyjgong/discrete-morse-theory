import gudhi
import random
from typing import Dict, Tuple, List, Optional
from manim import *
import numpy as np

class SimplicialComplex:
    """
    Wrapper around GUDHI SimplexTree for managing simplicial complexes.
    """
    def __init__(self):
        self.st = gudhi.SimplexTree()
    
    def add_simplex(self, simplex: Tuple, filtration: float = 0.0):
        # simplex is a tuple of vertices (e.g., ('A','B'))
        # GUDHI expects a list or tuple of vertices, filtration value
        self.st.insert(simplex, filtration=filtration)
    
    def get_simplices(self, dim: int) -> List[Tuple]:
        return [tuple(s[0]) for s in self.st.get_skeleton(dim) if len(s[0]) == dim+1]
    
    def compute_f_vector(self) -> List[int]:
        # f-vector: f0 = #vertices, f1 = #edges, ...
        max_dim = self.st.dimension()
        f_vector = []
        for d in range(max_dim+1):
            simplices = self.get_simplices(d)
            f_vector.append(len(simplices))
        return f_vector

class DiscreteMorseFunction:
    """
    Assign values to simplices and compute gradient vector fields.
    """
    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        # Dictionary mapping simplex (as a tuple) to a real value.
        self.values: Dict[Tuple, float] = {}
    
    def assign_values(self, values: Dict[Tuple, float]):
        for s in values:
            # Sort simplex tuple for consistency
            st = tuple(sorted(s))
            self.values[st] = values[s]
    
    def assign_random(self, low: float = 0.0, high: float = 1.0):
        for d in range(self.complex.st.dimension()+1):
            for s in self.complex.get_simplices(d):
                self.values[tuple(sorted(s))] = random.uniform(low, high)
    
    def compute_gradient_vector_field(self) -> 'GradientVectorField':
        gvf = GradientVectorField(self.complex, self.values)
        gvf.compute_pairs()
        return gvf
    
    def identify_critical_cells(self) -> List[Tuple]:
        gvf = self.compute_gradient_vector_field()
        return gvf.get_critical_cells()
    
    def perform_collapse(self):
        gvf = self.compute_gradient_vector_field()
        pairs = gvf.get_pairs()
        retained_simplices = set()
        critical = set(gvf.get_critical_cells())
        
        max_dim = self.complex.st.dimension()
        all_simplices = []
        for d in range(max_dim+1):
            all_simplices.extend(self.complex.get_simplices(d))
        
        # Mark non-critical that are involved in pairs for removal
        paired_simplices = set()
        for (s, t) in pairs:
            paired_simplices.add(tuple(sorted(t)))
        
        # Rebuild simplex tree with only retained simplices (critical + unpaired)
        new_complex = SimplicialComplex()
        for s in all_simplices:
            st = tuple(sorted(s))
            if st not in paired_simplices:
                new_complex.add_simplex(st, filtration=0.0)
        
        self.complex = new_complex

class GradientVectorField:
    """
    Stores pairings of simplices that define the gradient vector field.
    """
    def __init__(self, complex: SimplicialComplex, values: Dict[Tuple, float]):
        self.complex = complex
        self.values = values
        self.pairs: List[Tuple[Tuple, Tuple]] = []
    
    def compute_pairs(self):
        max_dim = self.complex.st.dimension()
        paired = set()
        
        for d in range(max_dim+1):
            for s in self.complex.get_simplices(d):
                s_sorted = tuple(sorted(s))
                if s_sorted in paired:
                    continue
                cofaces = self.get_cofaces(s_sorted)
                chosen = None
                for t in cofaces:
                    t_sorted = tuple(sorted(t))
                    if t_sorted not in paired and s_sorted not in paired:
                        if self.values[s_sorted] < self.values[t_sorted]:
                            chosen = t_sorted
                            break
                if chosen is not None:
                    self.pairs.append((s_sorted, chosen))
                    paired.add(s_sorted)
                    paired.add(chosen)
    
    def get_cofaces(self, simplex: Tuple) -> List[Tuple]:
        dim = len(simplex)-1
        cofaces = []
        coface_list = self.complex.st.get_cofaces(simplex, 1)
        for c in coface_list:
            cofaces.append(tuple(c[0]))
        return cofaces
    
    def get_pairs(self) -> List[Tuple[Tuple, Tuple]]:
        return self.pairs
    
    def get_critical_cells(self) -> List[Tuple]:
        paired_simplices = set()
        for (s, t) in self.pairs:
            paired_simplices.add(s)
            paired_simplices.add(t)

        critical = []
        max_dim = self.complex.st.dimension()
        for d in range(max_dim+1):
            for s in self.complex.get_simplices(d):
                st = tuple(sorted(s))
                if st not in paired_simplices:
                    critical.append(st)
        return critical

class TopologicalInvariants:
    @staticmethod
    def compute_betti_numbers(complex: SimplicialComplex) -> List[int]:
        st = complex.st
        st.compute_persistence()
        # Betti numbers can be derived by counting infinite intervals at each dimension
        betti = []
        max_dim = st.dimension()
        for d in range(max_dim+1):
            intervals = st.persistence_intervals_in_dimension(d)
            count = sum(1 for i in intervals if i[1] == float('inf'))
            betti.append(count)
        return betti
    
    @staticmethod
    def compute_euler_characteristic(complex: SimplicialComplex) -> int:
        f_vector = complex.compute_f_vector()
        chi = 0
        for i, f_i in enumerate(f_vector):
            chi += ((-1)**i)*f_i
        return chi

class PersistentHomology:
    @staticmethod
    def compute_persistence_diagram(complex: SimplicialComplex):
        st = complex.st
        st.compute_persistence()
        # Return the persistence diagram
        return st.persistence()
    
    @staticmethod
    def simplify_complex(complex: SimplicialComplex, dmf: DiscreteMorseFunction) -> SimplicialComplex:
        # Use dmf.perform_collapse()
        dmf.perform_collapse()
        return dmf.complex
    
    @staticmethod
    def compute_persistent_betti_numbers(complex: SimplicialComplex) -> List[Tuple[int, float, float]]:
        # Compute intervals and return (dimension, birth, death)
        st = complex.st
        st.compute_persistence()
        intervals = []
        for interval in st.persistence_intervals_in_dimension(0):
            intervals.append((0, interval[0], interval[1]))
        return intervals

class Visualization:
    @staticmethod
    def visualize_complex(complex: SimplicialComplex, vertex_positions: Dict[str, np.ndarray] = None):
        """
        Visualize the simplicial complex using Manim.
        
        Parameters
        ----------
        complex : SimplicialComplex
            The simplicial complex to visualize.
        vertex_positions : Dict[str, np.ndarray], optional
            A dictionary mapping vertex labels to 3D coordinates (e.g. {'A': np.array([0,0,0])}).
            If not provided, vertices will be arranged in a circle.
        """
        # Extract vertices
        all_vertices = complex.get_simplices(0)
        vertex_labels = [v[0] for v in all_vertices]  # Each vertex simplex is like ('A',)
        
        # If no positions given, arrange vertices in a circle
        if vertex_positions is None:
            n = len(vertex_labels)
            radius = 3.0
            vertex_positions = {}
            for i, v in enumerate(vertex_labels):
                angle = 2*np.pi*i/n
                vertex_positions[v] = np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
        
        # Extract edges (1-simplices)
        edges = complex.get_simplices(1)
        
        class ComplexScene(Scene):
            def construct(self):
                # Create vertex dots and labels
                vertex_mobs = {}
                for v in vertex_labels:
                    dot = Dot(point=vertex_positions[v], color=BLUE)
                    label = Text(v).scale(0.5).next_to(dot, DOWN*0.3)
                    vertex_mobs[v] = (dot, label)
                    self.play(GrowFromCenter(dot))
                    self.play(Write(label))
                
                # Draw edges
                for e in edges:
                    start, end = e
                    line = Line(vertex_positions[start], vertex_positions[end], color=WHITE)
                    self.play(Create(line))
        
        # Run the scene
        ComplexScene().render()

    @staticmethod
    def visualize_gradient_field(complex: SimplicialComplex, gvf: GradientVectorField, vertex_positions: Dict[str, np.ndarray] = None):
        """
        Visualize the gradient vector field using Manim.
        
        Parameters
        ----------
        complex : SimplicialComplex
            The simplicial complex to visualize.
        gvf : GradientVectorField
            The gradient vector field associated with a discrete Morse function.
        vertex_positions : Dict[str, np.ndarray], optional
            A dictionary mapping vertex labels to 3D coordinates.
            If not provided, vertices will be arranged in a circle.
        """
        # Extract vertices and edges
        all_vertices = complex.get_simplices(0)
        vertex_labels = [v[0] for v in all_vertices]
        
        if vertex_positions is None:
            n = len(vertex_labels)
            radius = 3.0
            vertex_positions = {}
            for i, v in enumerate(vertex_labels):
                angle = 2*np.pi*i/n
                vertex_positions[v] = np.array([radius*np.cos(angle), radius*np.sin(angle), 0])
        
        edges = complex.get_simplices(1)
        faces = complex.get_simplices(2)
        
        # Helper functions to get simplex centroid
        def centroid(simplex: Tuple[str]) -> np.ndarray:
            coords = [vertex_positions[x] for x in simplex]
            return sum(coords)/len(coords)
        
        pairs = gvf.get_pairs()
        critical_cells = gvf.get_critical_cells()
        
        # Create a scene
        class GradientFieldScene(Scene):
            def construct(self):
                vertex_mobs = {}
                # Draw vertices
                for v in vertex_labels:
                    dot = Dot(point=vertex_positions[v], color=BLUE)
                    label = Text(v).scale(0.5).next_to(dot, DOWN*0.3)
                    vertex_mobs[v] = dot
                    self.play(GrowFromCenter(dot))
                    self.play(Write(label))
                
                # Draw edges
                edge_mobs = {}
                for e in edges:
                    line = Line(vertex_positions[e[0]], vertex_positions[e[1]], color=WHITE)
                    edge_mobs[e] = line
                    self.play(Create(line))
                
                # Draw faces (if any)
                face_mobs = {}
                for f in faces:
                    polygon = Polygon(*[vertex_positions[x] for x in f], color=GREEN, fill_opacity=0.3)
                    face_mobs[f] = polygon
                    self.play(FadeIn(polygon))
                
                for (sigma, tau) in pairs:
                    start = centroid(sigma)
                    end = centroid(tau)
                    arrow = Arrow(start, end, buff=0.1, color=YELLOW)
                    self.play(GrowArrow(arrow))
                
                # Highlight critical cells
                # Critical vertices, edges, or faces. Change color or indicate them
                for c in critical_cells:
                    c_sorted = tuple(sorted(c))
                    dim = len(c)-1
                    if dim == 0:
                        # critical vertex
                        v = c[0]
                        self.play(Indicate(vertex_mobs[v], scale_factor=1.5, color=ORANGE))
                    elif dim == 1:
                        # critical edge
                        # Edges are stored as tuples, ensure sorted to match keys
                        e = c_sorted
                        if e in edge_mobs:
                            self.play(Indicate(edge_mobs[e], scale_factor=1.5, color=ORANGE))
                    else:
                        # critical face
                        f = c_sorted
                        if f in face_mobs:
                            self.play(Indicate(face_mobs[f], scale_factor=1.5, color=ORANGE))
        
        GradientFieldScene().render()
