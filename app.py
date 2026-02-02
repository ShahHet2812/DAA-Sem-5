import streamlit as st
import networkx as nx
from collections import deque
from streamlit_agraph import agraph, Node, Edge, Config


def bfs_steps(graph, start):
    visited = set()
    queue = deque([start])
    steps = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            steps.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return steps


def dfs_steps(graph, start):
    visited = set()
    steps = []

    def dfs(node):
        visited.add(node)
        steps.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return steps


st.set_page_config(page_title="Graph Traversal Visualizer", layout="wide")
st.title("Graph Traversal Visualizer (BFS & DFS)")
st.caption("Manual step-by-step traversal with fixed graph layout")


if "vertices" not in st.session_state:
    st.session_state.vertices = ""
if "edges" not in st.session_state:
    st.session_state.edges = ""
if "start" not in st.session_state:
    st.session_state.start = ""
if "algorithm" not in st.session_state:
    st.session_state.algorithm = "BFS"
if "steps" not in st.session_state:
    st.session_state.steps = []
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "graph" not in st.session_state:
    st.session_state.graph = None
if "edge_list" not in st.session_state:
    st.session_state.edge_list = []
if "positions" not in st.session_state:
    st.session_state.positions = {}


st.sidebar.header("Graph Configuration")

st.session_state.vertices = st.sidebar.text_input(
    "Vertices (comma-separated)",
    value=st.session_state.vertices
)

st.session_state.edges = st.sidebar.text_area(
    "Edges (one per line: A B)",
    value=st.session_state.edges
)

st.session_state.start = st.sidebar.text_input(
    "Starting Vertex",
    value=st.session_state.start
)

st.session_state.algorithm = st.sidebar.radio(
    "Traversal Algorithm",
    ["BFS", "DFS"],
    index=0 if st.session_state.algorithm == "BFS" else 1
)

generate_btn = st.sidebar.button("Generate Graph")


if generate_btn:
    vertices = [v.strip() for v in st.session_state.vertices.split(",") if v.strip()]
    if not vertices:
        st.error("Vertex list cannot be empty.")
        st.stop()

    if len(vertices) != len(set(vertices)):
        st.error("Duplicate vertices detected.")
        st.stop()

    edges = []
    edge_set = set()

    for line in st.session_state.edges.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            st.error(f"Invalid edge format: '{line}'.")
            st.stop()

        u, v = parts

        if u not in vertices or v not in vertices:
            st.error(f"Edge ({u}, {v}) contains an undefined vertex.")
            st.stop()

        if u == v:
            st.error(f"Self-loop detected at vertex '{u}'.")
            st.stop()

        key = tuple(sorted((u, v)))
        if key in edge_set:
            st.error(f"Duplicate edge detected: ({u}, {v}).")
            st.stop()

        edge_set.add(key)
        edges.append((u, v))

    if st.session_state.start not in vertices:
        st.error("Starting vertex must exist in the vertex list.")
        st.stop()

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    adjacency_list = {
        node: sorted(list(G.neighbors(node)))
        for node in G.nodes
    }

    if st.session_state.algorithm == "BFS":
        st.session_state.steps = bfs_steps(adjacency_list, st.session_state.start)
    else:
        st.session_state.steps = dfs_steps(adjacency_list, st.session_state.start)

    st.session_state.current_step = 0
    st.session_state.graph = G
    st.session_state.edge_list = edges

    pos = nx.spring_layout(G, seed=42)
    st.session_state.positions = {
        node: (pos[node][0] * 500, pos[node][1] * 500)
        for node in G.nodes
    }


if st.session_state.graph is not None and st.session_state.steps and st.session_state.positions:
    st.subheader("Graph Visualization")

    visited = set(st.session_state.steps[: st.session_state.current_step + 1])
    current = st.session_state.steps[st.session_state.current_step]

    nodes = []
    for v in st.session_state.graph.nodes:
        x, y = st.session_state.positions[v]
        nodes.append(
            Node(
                id=v,
                label=v,
                x=x,
                y=y,
                size=22,
                color="#2ECC71" if v in visited else "#2F80ED",
                fixed=True
            )
        )

    edges_draw = [
        Edge(source=u, target=v, color="#666666", width=2)
        for u, v in st.session_state.edge_list
    ]

    config = Config(
        width=600,
        height=450,
        directed=False,
        physics=False,
        hierarchical=False,
        backgroundColor="#FFFFFF",
        font={"color": "#FFFFFF", "size": 16}
    )

    agraph(nodes=nodes, edges=edges_draw, config=config)

    st.write(f"Current node: {current}")

    col1, col2, col3 = st.columns(3)

    with col2:
        if st.button("Next Step"):
            if st.session_state.current_step < len(st.session_state.steps) - 1:
                st.session_state.current_step += 1
                st.rerun()

    st.subheader("Traversal Order So Far")
    st.success(" → ".join(st.session_state.steps[: st.session_state.current_step + 1]))

    st.subheader("Time and Space Complexity")

    V = st.session_state.graph.number_of_nodes()
    E = st.session_state.graph.number_of_edges()

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"BFS Time: O(V + E) = O({V} + {E})")
        st.write(f"BFS Space: O(V) = O({V})")

    with col2:
        st.write(f"DFS Time: O(V + E) = O({V} + {E})")
        st.write(f"DFS Space: O(V) = O({V})")
