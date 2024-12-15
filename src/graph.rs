use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io;
use petgraph::Graph;
use petgraph::graph::NodeIndex;
use io::Write;

// graph algorithm
pub(crate) fn build_similarity_graph(
    file_path: &str,
    features: &[usize],
    threshold: f64, // Similarity threshold
) -> Result<Graph<String, f64>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut graph = Graph::<String, f64>::new();
    let mut nodes = Vec::new();
    let mut feature_data = Vec::new();

    for record in reader.records() {
        let record = record?;
        let country = record.get(0).unwrap_or("").to_string();
        nodes.push(country);
        let features_row: Vec<f64> = features
            .iter()
            .filter_map(|&idx| record.get(idx).and_then(|val| val.parse::<f64>().ok()))
            .collect();
        feature_data.push(features_row);
    }

    // Add nodes to the graph
    let node_indices: Vec<_> = nodes
        .iter()
        .map(|country| graph.add_node(country.clone()))
        .collect();

    // Calculate pairwise similarity and add edges
    for i in 0..feature_data.len() {
        for j in (i + 1)..feature_data.len() {
            let similarity = calculate_similarity(&feature_data[i], &feature_data[j]);
            if similarity >= threshold {
                graph.add_edge(node_indices[i], node_indices[j], similarity);
            }
        }
    }

    Ok(graph)
}

// Calculate similarity between two feature vectors
fn calculate_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2).map(|(x, y)| x * y).sum();
    let magnitude1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if magnitude1 > 0.0 && magnitude2 > 0.0 {
        dot_product / (magnitude1 * magnitude2)
    } else {
        0.0
    }
}

// Perform graph clustering and identify representatives
pub(crate) fn cluster_graph(graph: &Graph<String, f64>, _k: usize) -> HashMap<usize, String> {
    use petgraph::unionfind::UnionFind;

    // Determine connected components
    let mut uf = UnionFind::new(graph.node_count());
    for edge in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge).unwrap();
        uf.union(a.index(), b.index());
    }

    // Map each component to its nodes
    let mut clusters: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for node in graph.node_indices() {
        let component_id = uf.find(node.index());
        clusters.entry(component_id).or_default().push(node);
    }

    // Select a representative for each cluster
    let mut representatives = HashMap::new();
    for (cluster_id, nodes) in clusters {
        if let Some(representative) = select_representative(&graph, &nodes) {
            representatives.insert(cluster_id, graph[representative].clone());
        }
    }

    representatives
}

// Select a representative node based on centrality
fn select_representative(
    graph: &Graph<String, f64>,
    nodes: &[NodeIndex],
) -> Option<NodeIndex> {
    nodes
        .iter()
        .max_by_key(|&&node| graph.edges(node).count())
        .cloned()
}

// Visualize Graph Algorithm
pub(crate) fn export_graph_to_csv(
    graph: &Graph<String, f64>,
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    // Open the output file for writing
    let mut file = File::create(output_file)?;

    // Write the CSV header
    writeln!(file, "Source,Target,Weight")?;

    // Iterate over the edges in the graph
    for edge in graph.edge_indices() {
        let (source, target) = graph.edge_endpoints(edge).unwrap();
        let weight = graph.edge_weight(edge).unwrap();

        // Write each edge as a row in the CSV file
        writeln!(
            file,
            "{}, {}, {:.6}",
            graph[source],
            graph[target],
            weight
        )?;
    }

    Ok(())
}
