from .agent import create_finance_agent
from .chains import create_classifier_chain, create_combine_chain, decompose_complex_query
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import pandas as pd

# Load LLM
llm = ChatGoogleGenerativeAI(
    model=Config.MODEL_NAME,
    temperature=Config.TEMPERATURE,
    top_k=Config.TOP_K,
    top_p=Config.TOP_P
)

# Create agent
agent = create_finance_agent(llm)

# Create classifier chain
classifier_chain = create_classifier_chain(llm)

# Create combine chain
combine_chain = create_combine_chain(llm)

app = FastAPI(title="Finance Agent API", description="API for financial analysis and stock market queries")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    is_complex: bool
    reasoning: str = None

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a financial query and return the answer."""
    try:
        query = request.query
        
        # Classify the query
        classification_result = classifier_chain.invoke({"query": query})
        
        print(f"Classification result is complex: {classification_result.is_complex}")
        print(f"Classification result reasoning: {classification_result.reasoning}")
        
        if classification_result.is_complex:
            # Decompose complex query
            decomposed_queries = decompose_complex_query(query, llm)
            
            # Process each decomposed query with agent
            results = []
            for sub_query in decomposed_queries:
                agent_result = agent.invoke({"messages": [{"role": "user", "content": sub_query}]})
                results.append(agent_result)
            
            # Format results into string
            formatted_results = ""
            for i, result in enumerate(results, 1):
                formatted_results += f"Kết quả {i}:\n{result.get('output', str(result))}\n\n"
            
            # Combine results using the chain
            combined_result = combine_chain.invoke({
                "original_query": query,
                "results": formatted_results
            })
            
            return QueryResponse(
                answer=combined_result.combined_answer,
                is_complex=True,
                reasoning=classification_result.reasoning
            )
        
        else:
            # Process simple query directly
            result = agent.invoke({"messages": [{"role": "user", "content": query}]})
            
            # Handle different response formats
            if 'structured_response' in result:
                # Case 1: return_direct=False - formatted response
                print(f"Case 1: return_direct=False")
                answer = result['structured_response'].content
            else:
                # Case 2: return_direct=True - raw tool data
                try:
                    print(f"Case 2: return_direct=True")
                    outer = json.loads(result['messages'][-1].content)
                    all_rows = []

                    for symbol, inner_json_str in outer.items():
                        records = json.loads(inner_json_str)
                        for r in records:
                            r["symbol"] = symbol
                        all_rows.extend(records)

                    df = pd.DataFrame(all_rows)
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    
                    # Convert DataFrame to readable format
                    answer = df.to_string(index=False)
                except:
                    # Fallback to string representation
                    print(f"Fallback to string representation")
                    answer = str(result.get('output', result))
            
            return QueryResponse(
                answer=answer,
                is_complex=False,
                reasoning=classification_result.reasoning
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Finance Agent API",
        "description": "API for financial analysis and stock market queries",
        "endpoints": {
            "/query": "POST - Process financial queries",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)