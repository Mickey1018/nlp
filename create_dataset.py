from openpyxl import load_workbook
import pandas as pd


def get_dataset(excel, col="Question", df_dataset=None):

    # Load the Excel file
    workbook = load_workbook(excel)

    # Get the sheet names
    sheet_names = workbook.sheetnames

    try:
        if not df_dataset:
            df_dataset = pd.Series()
    except:
        pass

    for sheet_name in sheet_names:

        if sheet_name.lower() == "summary":
            continue
        
        # get intent from sheet name
        intent = sheet_name.split(".")[-1]
        intent = "_".join(intent.split())
        intent = intent.lower()
        
        try:
            df_temp = pd.read_excel(excel, sheet_name=sheet_name)
            df_temp = df_temp[df_temp[col].notnull()]

            # get data
            texts = df_temp[col].tolist()
            texts = [text.replace("\n", "") for text in texts]
            text_col = pd.Series(texts)
            label_col = pd.Series(intent for _ in range(len(texts)))

            df_dataset_temp = pd.DataFrame({"text": text_col, "label": label_col})

            df_dataset = pd.concat([df_dataset, df_dataset_temp]).reset_index(drop=True)
            
        except Exception as e:
            print(excel, sheet_name)
            print(e)
            pass
        
    
    df_dataset = df_dataset.reset_index(drop=True)

    return df_dataset


if __name__ == "__main__":
    df = get_dataset("data/nlu/Intent_Chinese_20231109.xlsx")
    df = get_dataset("data/nlu/Intent_Simplified_Chinese_20231109.xlsx", df_dataset=df)
    df = get_dataset("data/nlu/Intent_English_20231109.xlsx", df_dataset=df)
    
    df.to_excel("data/nlu/dataset.xlsx")