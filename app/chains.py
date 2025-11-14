# =============Classifer chain=============
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel

class QueryClassification(BaseModel):
    is_complex: bool = Field(description="True nếu câu hỏi phức tạp, False nếu đơn giản")
    reasoning: str = Field(description="Lý do phân loại")

class ComplexQuery(BaseModel):
    query: str = Field(description="Câu hỏi phức tạp cần phân tích và trả lời")

# Create classifier chain
def create_classifier_chain(llm: BaseChatModel):
    """Tạo chain để phân loại câu hỏi."""
    
    # Parser để chuyển đổi output thành Pydantic model
    parser = PydanticOutputParser(pydantic_object=QueryClassification)
    
    # Prompt template cho classifier
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia phân loại câu hỏi tài chính.
        Nhiệm vụ: Phân loại câu hỏi thành True (phức tạp) hoặc False (đơn giản).
        
        Câu hỏi ĐƠN GIẢN (False): Chỉ cần 1 công cụ để trả lời (ví dụ: giá cổ phiếu, thông tin cơ bản)
        Câu hỏi PHỨC TẠP (True): Cần nhiều công cụ hoặc phân tích sâu (ví dụ: so sánh, xu hướng, dự đoán)
        
        {format_instructions}"""),
        ("human", "Phân loại câu hỏi sau: {query}")
    ])
    
    # Format instructions cho parser
    classifier_prompt = classifier_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Tạo chain
    classifier_chain = classifier_prompt | llm | parser
    
    return classifier_chain

# =============Decompose complex query=============
class QueryDecomposition(BaseModel):
    sub_queries: list[str] = Field(description="Danh sách các câu hỏi con được phân tách từ câu hỏi phức tạp")
    reasoning: str = Field(description="Lý do phân tách câu hỏi")

def create_decomposition_chain(llm: BaseChatModel):
    """Tạo chain để phân tách câu hỏi phức tạp thành các câu hỏi con."""
    
    # Parser để chuyển đổi output thành Pydantic model
    parser = PydanticOutputParser(pydantic_object=QueryDecomposition)
    
    # Prompt template cho decomposition
    decomposition_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia phân tích câu hỏi tài chính.
        Nhiệm vụ: Phân tách câu hỏi phức tạp thành các câu hỏi con đơn giản hơn.
        
        Nguyên tắc phân tách:
        - Mỗi câu hỏi con chỉ cần 1 công cụ để trả lời
        - Các câu hỏi con phải có thứ tự logic
        - Đảm bảo câu hỏi con bao phủ toàn bộ câu hỏi gốc
        - Sử dụng mã cổ phiếu cụ thể nếu có
        - Nếu có thể dùng 1 công cụ cho nhiều công ty (như danh sách), hãy gộp lại
        
        Ví dụ:
        Câu hỏi: "So sánh hiệu suất VIC và VHM trong 3 tháng qua"
        Phân tách thành:
        1. "Tính SMA của VIC, VHM trong 3 tháng qua"
        2. "Tính RSI của VIC, VHM trong 3 tháng qua"
        Lý do: Câu hỏi yêu cầu tính toán 2 chỉ số khác nhau, cần phân tách để giải quyết.
        
        Câu hỏi: "Danh sách cổ đông lớn của VCB và TCB"
        Không phân tách câu hỏi
        Lý do: Công cụ cho phép tính toán trên nhiều mã cổ phiếu, không cần phân tách.
        
        Câu hỏi: "So sánh khối lượng giao dịch của VIC với HPG trong 2 tuần gần đây"
        Không phân tách câu hỏi
        Lý do: Công cụ cho phép tính toán trên nhiều mã cổ phiếu, không cần phân tách.

        Câu hỏi: "Danh sách ban lãnh đạo đang làm việc của VCB và Các công ty con thuộc VCB"
        Phân tách thành:
        1. "Danh sách ban lãnh đạo đang làm việc của VCB"
        2. "Danh sách các công ty con thuộc VCB"
        Lý do: Câu hỏi yêu cầu sử dụng 2 công cụ khác nhau, cần phân tách để giải quyết.
        
        Câu hỏi: "Tính cho tôi SMA9 và SMA20 của mã VIC trong 2 tháng với timeframe 1d"
        Phân tách thành:
        1. "Tính SMA9 của VIC trong 2 tháng với timeframe 1d"
        2. "Tính SMA20 của VIC trong 2 tháng với timeframe 1d"
        Lý do: Câu hỏi yêu cầu tính toán 1 chỉ số 2 lần với config khác nhau, cần phân tách để giải quyết.
        
        {format_instructions}"""),
        ("human", "Phân tách câu hỏi sau: {query}")
    ])
    
    # Format instructions cho parser
    decomposition_prompt = decomposition_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Tạo chain
    decomposition_chain = decomposition_prompt | llm | parser
    
    return decomposition_chain

def decompose_complex_query(query, llm: BaseChatModel):
    """Phân tách câu hỏi phức tạp thành các câu hỏi con."""
    decomposition_chain = create_decomposition_chain(llm)
    result = decomposition_chain.invoke({"query": query})
    return result.sub_queries

#==========Combine result==========
class CombinedResult(BaseModel):
    combined_answer: str = Field(description="Câu trả lời tổng hợp từ nhiều kết quả con")
    original_query: str = Field(description="Câu hỏi gốc")

def create_combine_chain(llm: BaseChatModel):
    """Tạo chain để kết hợp nhiều kết quả thành một câu trả lời tổng hợp."""
    
    # Parser để chuyển đổi output thành Pydantic model
    parser = PydanticOutputParser(pydantic_object=CombinedResult)
    
    # Prompt template cho combining results
    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia phân tích tài chính.
        Nhiệm vụ: Kết hợp nhiều kết quả từ các câu hỏi con thành một câu trả lời tổng hợp, mạch lạc.
        
        Nguyên tắc kết hợp:
        - Tổng hợp thông tin từ tất cả các kết quả
        - Tạo câu trả lời mạch lạc, dễ hiểu
        - So sánh và phân tích nếu có nhiều mã cổ phiếu
        - Đưa ra kết luận và khuyến nghị nếu phù hợp
        - Sử dụng tiếng Việt và định dạng dễ đọc
        
        {format_instructions}"""),
        ("human", """Câu hỏi gốc: {original_query}
        
        Các kết quả từ câu hỏi con:
        {results}
        
        Hãy kết hợp các kết quả trên thành một câu trả lời tổng hợp.""")
    ])
    
    # Format instructions cho parser
    combine_prompt = combine_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Tạo chain
    combine_chain = combine_prompt | llm | parser
    
    return combine_chain