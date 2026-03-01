from main.MAS import MAS

while 1 : 
    user_query = input("actions needed : ")
    dataset_path = input("Dataset path : ")
    response = MAS.print_response(
        f"actions needed : {user_query} , dataset path : {dataset_path}",
        stream=True
    )
