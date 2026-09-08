import fitz
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @staticmethod
    def linearize_markdown_table(markdown_table: str, parent_id: str, page_num: int):
        lines = markdown_table.strip().split('\n')
        
        if len(lines) < 3: 
            return []

        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        child_chunks = []
        
        for row in lines[2:]:
            cells = [c.strip() for c in row.split('|') if c.strip()]
            
            if len(cells) == len(headers):
                row_elements = [f"{headers[i]}: {cells[i]}" for i in range(len(headers))]
                dense_row_string = " | ".join(row_elements)
                
                # Create a LangChain Document right here for the child chunk
                metadata = {
                    "source": parent_id,
                    "page": page_num,
                    "type": "table_row"
                }
                child_chunks.append(Document(page_content=dense_row_string, metadata=metadata))
                
        return child_chunks

    def loader(self):
        doc = fitz.open(self.file_path)
        documents = []
        source_name = self.file_path.split("/")[-1]
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            tables = page.find_tables()
            for tab in tables:
                # Convert the detected table to a pandas dataframe, then to a markdown string
                df = tab.to_pandas()
                markdown_table = df.to_markdown(index=False)
                
                # Pass it to your linearize function
                table_row_docs = self.linearize_markdown_table(
                    markdown_table=markdown_table, 
                    parent_id=source_name, 
                    page_num=page_num + 1
                )
                documents.extend(table_row_docs)
            
            # ---------------------------------------------------
            # ORIGINAL: STANDARD TEXT BLOCK EXTRACTION
            # ---------------------------------------------------
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if len(block) < 7:
                    continue
                    
                text = block[4]
                block_type = block[6]
                
                if block_type != 0 or not isinstance(text, str) or not text.strip():
                    continue
                    
                clean_text = text.replace("-\n", "").replace("\n", " ").strip()
                
                metadata = {
                    "source": source_name,
                    "page": page_num + 1,
                    "type": "text_block"
                }
                
                documents.append(Document(page_content=clean_text, metadata=metadata))
                
        doc.close()
        return documents