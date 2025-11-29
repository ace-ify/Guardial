"""
Fallback vector_db implementation using ChromaDB's built-in embeddings.
This version works with Python 3.13 and doesn't require sentence-transformers.
"""
import chromadb
import logging

logger = logging.getLogger(__name__)

# Initialize ChromaDB client with default embedding function
# ChromaDB will use its built-in sentence transformer embeddings
client = chromadb.PersistentClient(path="./chroma_db")

# In-memory cache for collections
collections = {}

# Default ISR (Information Source Retrieval) threshold configuration
DEFAULT_ISR_THRESHOLD = 0.40
ISR_CONFIG = {
    "threshold": DEFAULT_ISR_THRESHOLD,
    "min_threshold": 0.1,
    "max_threshold": 0.95,
    "distance_metric": "l2",
}

def get_or_create_collection(name="hallucination_auditor"):
    """Gets a ChromaDB collection or creates it if it doesn't exist."""
    if name in collections:
        return collections[name]
    
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    collections[name] = collection
    logger.info(f"Collection '{name}' retrieved/created successfully")
    return collection

def add_documents_to_collection(documents, collection_name="hallucination_auditor"):
    """Adds a list of documents to a specified ChromaDB collection."""
    collection = get_or_create_collection(collection_name)
    
    # Generate IDs for the documents
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add the documents - ChromaDB will handle embeddings automatically
    collection.add(
        documents=documents,
        ids=ids
    )
    logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
    return True

def distance_to_similarity(distance, metric="cosine"):
    """
    Converts distance to similarity score (0-1 range).
    
    Args:
        distance: The distance value from vector search
        metric: The distance metric used ('l2', 'cosine', etc.)
    
    Returns:
        similarity: A score between 0 (completely different) and 1 (identical)
    """
    if metric == "cosine":
        # Cosine distance is 1 - cosine_similarity, so we invert it
        # Distance ranges from 0 (identical) to 2 (opposite)
        similarity = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, similarity))
    elif metric == "l2":
        # For L2 distance: use exponential decay
        import math
        similarity = math.exp(-distance * 0.5)
        return max(0.0, min(1.0, similarity))
    else:
        # Default: exponential decay
        import math
        return max(0.0, min(1.0, math.exp(-distance * 0.5)))

def calculate_isr_score(query_results):
    """
    Calculates the ISR (Information Source Retrieval) score from query results.
    
    Returns:
        dict: ISR score metrics including best match score and explanation
    """
    if not query_results or not query_results.get('distances') or not query_results['distances'][0]:
        return {
            "score": 0.0,
            "confidence": "none",
            "explanation": "No matching documents found in the knowledge base."
        }
    
    # Get the best (closest) match distance
    best_distance = query_results['distances'][0][0]
    best_doc = query_results['documents'][0][0] if query_results.get('documents') else None
    
    # Convert distance to similarity score
    similarity_score = distance_to_similarity(best_distance, "cosine")
    
    # Determine confidence level
    if similarity_score >= 0.75:
        confidence = "very_high"
        explanation = "Query directly matches knowledge base content."
    elif similarity_score >= 0.55:
        confidence = "high"
        explanation = "Query is closely related to knowledge base content."
    elif similarity_score >= 0.35:
        confidence = "moderate"
        explanation = "Query has moderate relevance to knowledge base."
    elif similarity_score >= 0.20:
        confidence = "low"
        explanation = "Query has limited relevance to knowledge base."
    else:
        confidence = "very_low"
        explanation = "Query appears to be outside knowledge base scope."
    
    return {
        "score": similarity_score,
        "distance": best_distance,
        "confidence": confidence,
        "explanation": explanation,
        "matched_document": best_doc[:200] + "..." if best_doc and len(best_doc) > 200 else best_doc
    }

def query_collection(query, collection_name="hallucination_auditor", n_results=5):
    """Queries a ChromaDB collection and returns the most similar documents."""
    collection = get_or_create_collection(collection_name)
    
    # Query the collection - ChromaDB handles embedding automatically
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    logger.info(f"Query executed: '{query[:50]}...' - Found {len(results.get('documents', [[]])[0])} results")
    
    return results

def check_isr_threshold(query, custom_threshold=None):
    """
    Checks if a query meets the ISR threshold for allowing a response.
    
    Args:
        query: The user's query text
        custom_threshold: Optional custom threshold (overrides default)
        
    Returns:
        dict: Decision info including whether to allow/block and ISR metrics
    """
    threshold = custom_threshold if custom_threshold is not None else ISR_CONFIG["threshold"]
    
    # Ensure threshold is within valid range
    threshold = max(ISR_CONFIG["min_threshold"], min(ISR_CONFIG["max_threshold"], threshold))
    
    # Query the collection
    results = query_collection(query, n_results=1)
    
    # Calculate ISR score
    isr_metrics = calculate_isr_score(results)
    
    # Make decision based on threshold
    allow_response = isr_metrics["score"] >= threshold
    
    decision = {
        "allow": allow_response,
        "decision": "Allowed" if allow_response else "Blocked",
        "isr_score": isr_metrics["score"],
        "threshold": threshold,
        "confidence": isr_metrics["confidence"],
        "explanation": isr_metrics["explanation"],
        "matched_document": isr_metrics.get("matched_document"),
        "reason": f"ISR score ({isr_metrics['score']:.2f}) {'meets' if allow_response else 'below'} threshold ({threshold:.2f})"
    }
    
    logger.info(f"ISR Check: {decision['decision']} - Score: {isr_metrics['score']:.2f}, Threshold: {threshold:.2f}")
    
    return decision

def set_isr_threshold(new_threshold):
    """Updates the ISR threshold configuration."""
    if ISR_CONFIG["min_threshold"] <= new_threshold <= ISR_CONFIG["max_threshold"]:
        ISR_CONFIG["threshold"] = new_threshold
        logger.info(f"ISR threshold updated to {new_threshold:.2f}")
        return True
    else:
        logger.warning(f"Invalid threshold {new_threshold}. Must be between {ISR_CONFIG['min_threshold']} and {ISR_CONFIG['max_threshold']}")
        return False

def get_isr_config():
    """Returns the current ISR configuration."""
    return ISR_CONFIG.copy()

def clear_collection(collection_name="hallucination_auditor"):
    """Deletes all items from a collection."""
    if collection_name in collections:
        client.delete_collection(name=collection_name)
        del collections[collection_name]
        logger.info(f"Collection '{collection_name}' cleared successfully")
        return True
    return False
