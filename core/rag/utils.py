from pathlib import Path
import aiofiles
async def read_txt_file(file_str:str)->str:
    file_path=Path(file_str)
    results = file_path.rglob('*.txt')
    all_data=[]
    for file in results:
        async with aiofiles.open(file,mode='r') as f:
            data=await f.read()
            all_data.append(data)
    combined_text="/n".join(all_data)
    return combined_text
        
        
    
    