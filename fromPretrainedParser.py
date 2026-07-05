f_list = open('grep_file_list.txt', 'r')
all_prjs = f_list.readlines()

out_prj_modelList = open('prj_modelList.csv', 'w')

out_prj_file_model = open('prj_file_model.csv', 'w')

for pp in all_prjs:
    with open('grepResults/' + pp.strip(), 'r') as ff:
        #print(pp.strip())
        lines = ff.readlines()
        prj_name = pp.replace('£sep£', '/').replace('.txt', '')

        modelNameList = []
        for ll in lines:
            filePath = ll.split(':')[0]
            fromPre = ll.split(':')[1]
            if('.from_pretrained(' in fromPre):
                mName = fromPre.split('.from_pretrained(')[1]

                if("," in mName):
                    #print(mName.strip())
                    splitName = mName.split(",")
                    name = splitName[0]
                    name = name.strip().split(")")[0]
                    #print(name.strip().split(")")[0])
                elif(")" in mName):
                    name = mName.split(")")[0]
                else:
                    name = mName
                # \" f'
                name = name.strip()
                modelNameList.append(name)

                out_prj_file_model.write(prj_name.strip() + ',' + filePath + ',' + name.strip() + '\n')

        out_prj_modelList.write(prj_name.strip() + ',' + str(modelNameList).strip() + '\n')

out_prj_modelList.close()
out_prj_file_model.close()
