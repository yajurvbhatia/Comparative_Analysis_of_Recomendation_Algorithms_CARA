import snap
import networkx as nx

def initGraph():
    # load the facebook graph from snap
    G1 = snap.LoadEdgeList(snap.PNGraph, "Dataset/facebook_combined.txt", 0, 1)
    # print("G1: Nodes %d, Edges %d" % (G1.GetNodes(), G1.GetEdges()))

    # Create a Networkx graph from the snap graph by adding the nodees and edges
    nxG = nx.Graph()
    for node in G1.Nodes():
        nid = node.GetId()
        nxG.add_node(nid)
        print(node.feat)
        # nxG.nodes[nid]['features'] =
    for edge in G1.Edges():
        nxG.add_edge(edge.GetSrcNId(), edge.GetDstNId())
    nx.write_gpickle(nxG, 'sample_G')
    nxG = nx.read_gpickle('sample_G')
    # nx.draw_networkx(G)
    print("nxG: Nodes %d, Edges %d" % (nxG.number_of_nodes(), nxG.number_of_edges()))

initGraph()
