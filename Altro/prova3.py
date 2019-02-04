from Utilities.Utility import *


path = "../Results_CSV"
path = "C:/Users/Michele/Desktop/Grid Search/7000 Epoche/Grid_27_1_19/Risultati"
best_file, best_vl_error=analyze_result_csv(path,5)
print("Best_file ",best_file)
print("Best_vl_error ",best_vl_error)

with open("../Risultati_Prove/risultati.txt","a") as f:
    f.write("1000 epoche biblioteca : Best File %s\n"%(best_file))