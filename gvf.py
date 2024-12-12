from manim import *

class DiscreteMorseGradientField(Scene):
    def construct(self):
        # Define vertices of the simplicial complex
        vertices = {
            'A': np.array([-3, -1, 0]),
            'B': np.array([-1, 2, 0]),
            'C': np.array([1, -1, 0]),
            'D': np.array([3, 2, 0]),
        }

        # Define edges (1-simplices)
        edges = [
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'C'),
            ('B', 'D'),
            ('C', 'D'),
        ]

        # Define faces (2-simplices)
        faces = [
            ('A', 'B', 'C'),
            ('B', 'C', 'D'),
        ]

        # Create dots for vertices
        vertex_dots = {name: Dot(point=pos, color=BLUE) for name, pos in vertices.items()}
        vertex_labels = {name: Text(name).next_to(dot, DOWN) for name, dot in vertex_dots.items()}

        # Create lines for edges
        edge_lines = {}
        for edge in edges:
            start, end = edge
            line = Line(vertices[start], vertices[end], color=WHITE)
            edge_lines[edge] = line

        # Create polygons for faces
        face_polygons = {}
        for face in faces:
            points = [vertices[vertex] for vertex in face]
            polygon = Polygon(*points, color=GREEN, fill_opacity=0.5)
            face_polygons[face] = polygon

        # Add faces to the scene
        self.play(*[FadeIn(polygon) for polygon in face_polygons.values()])

        # Add edges to the scene
        self.play(*[Create(line) for line in edge_lines.values()])

        # Add vertices to the scene
        self.play(*[GrowFromCenter(dot) for dot in vertex_dots.values()])
        self.play(*[Write(label) for label in vertex_labels.values()])

        # Define the gradient vector field
        gradient_arrows = []

        # Vertex-Edge pairings (arrows from vertices to edges)
        vertex_edge_pairs = [
            ('C', ('A', 'C')),  # Vertex C paired with edge AC
            ('D', ('B', 'D')),  # Vertex D paired with edge BD
        ]

        for vertex, edge in vertex_edge_pairs:
            start = vertices[vertex]
            end = (vertices[edge[0]] + vertices[edge[1]]) / 2
            arrow = Arrow(start, end, buff=0.1, color=YELLOW)
            gradient_arrows.append(arrow)

        # Edge-Face pairings (arrows from edges to faces)
        edge_face_pairs = [
            (('B', 'C'), ('A', 'B', 'C')),  # Edge BC paired with face ABC
            (('C', 'D'), ('B', 'C', 'D')),  # Edge CD paired with face BCD
        ]

        for edge, face in edge_face_pairs:
            start = (vertices[edge[0]] + vertices[edge[1]]) / 2
            face_center = sum([vertices[vertex] for vertex in face]) / 3
            arrow = Arrow(start, face_center, buff=0.1, color=RED)
            gradient_arrows.append(arrow)

        # Add gradient arrows to the scene
        self.play(*[GrowArrow(arrow) for arrow in gradient_arrows])

        # Highlight critical simplices (unpaired simplices)
        # Critical vertices: 'A', 'B'
        critical_vertices = ['A', 'B']
        for name in critical_vertices:
            dot = vertex_dots[name]
            dot.set_color(ORANGE)
            self.play(Indicate(dot, scale_factor=1.5))

        # Critical edge: ('A', 'B')
        critical_edge = ('A', 'B')
        critical_line = edge_lines[critical_edge]
        critical_line.set_color(ORANGE)
        self.play(Indicate(critical_line, scale_factor=1.5))

        # Hold the final scene
        self.wait(2)
