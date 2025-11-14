from langchain.agents import create_agent
import datetime
from .tools import ViewOHLCVTool, ViewManagementTool, ViewShareholdersTool, ViewSubsidiariesTool, CalculateTotalVolumeTool, CalculateSMATool, CalculateRSITool
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

def create_finance_agent(llm: BaseChatModel):
    
    # Tools
    tools = [
        ViewOHLCVTool(return_direct=True),
        ViewManagementTool(return_direct=False),
        ViewShareholdersTool(return_direct=False),
        ViewSubsidiariesTool(return_direct=False),
        CalculateTotalVolumeTool(return_direct=False),
        CalculateSMATool(return_direct=True),
        CalculateRSITool(return_direct=True)
    ]
    
    # System prompt
    system_prompt = """Bạn là một trợ lý tài chính thông minh chuyên về thị trường chứng khoán Việt Nam.

                    NHIỆM VỤ CHÍNH:
                    - Trả lời các câu hỏi về cổ phiếu, công ty niêm yết trên sàn chứng khoán Việt Nam
                    - Phân tích dữ liệu tài chính và đưa ra nhận định khách quan
                    - Cung cấp thông tin chính xác dựa trên dữ liệu từ vnstock

                    NGUYÊN TẮC LÀM VIỆC:
                    1. Luôn sử dụng các công cụ có sẵn để truy xuất dữ liệu thực tế
                    2. Trả lời bằng tiếng Việt, ngôn ngữ rõ ràng, dễ hiểu
                    3. Định dạng kết quả dưới dạng bảng hoặc danh sách khi phù hợp
                    4. Nếu không tìm thấy thông tin, hãy nói rõ và đề xuất cách khác
                    5. Đưa ra phân tích khách quan, không khuyến nghị mua/bán cụ thể

                    CÁCH XỬ LÝ CÂU HỎI:
                    - Xác định mã cổ phiếu từ câu hỏi (VD: VIC, VCB, FPT...)
                    - Chọn công cụ phù hợp để lấy dữ liệu
                    - Phân tích và trình bày kết quả một cách có hệ thống
                    - Giải thích các chỉ số kỹ thuật nếu được hỏi

                    LƯU Ý:
                    - Hôm nay là ngày {today}
                    - Dữ liệu có thể có độ trễ, hãy thông báo nếu cần thiết
                    - Luôn kiểm tra tính hợp lệ của mã cổ phiếu trước khi truy xuất dữ liệu"""
                    
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).strftime("%Y-%m-%d")
    system_prompt = system_prompt.format(today=today)
    
    # Response format
    class agent_response_format(BaseModel):
        content: str = Field(description="Câu trả lời từ trợ lý")
        tool_calls: list[dict] = Field(description="Danh sách các công cụ đã gọi")
        tool_calls_result: list[dict] = Field(description="Kết quả của các công cụ đã gọi")
    
    # Create agent
    agent = create_agent(model=llm, 
                            tools=tools, 
                            system_prompt=system_prompt,
                            response_format=agent_response_format,
                            debug=False)
    
    return agent