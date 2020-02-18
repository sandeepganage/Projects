
def print_decision_tree(tree, feature_names, offset_unit='    '):
    left= tree.tree_.children_left
    right= tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    if feature_names is None:
        features  = ['f%d'%i for i in tree.tree_.feature]
    else:
        features  = [feature_names[i] for i in tree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0):
        offset = offset_unit*depth
        if (threshold[node] != -2):
            print(offset+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse (left, right, threshold, features,left[node],depth+1)
            print(offset+"} else {")
            if right[node] != -1:
                recurse (left, right, threshold, features,right[node],depth+1)
            print(offset+"}")
        else:
            #print(offset,value[node])

            #To remove values from node
            temp=str(value[node])
            mid=len(temp)//2
            tempx=[]
            tempy=[]
            cnt=0
            for i in temp:
                if cnt<=mid:
                    tempx.append(i)
                    cnt+=1
                else:
                    tempy.append(i)
                    cnt+=1
                val_yes=[]
                val_no=[]
                res=[]
                for j in tempx:
                    if j=="[" or j=="]" or j=="." or j==" ":
                        res.append(j)
                    else:
                        val_no.append(j)
                for j in tempy:
                    if j=="[" or j=="]" or j=="." or j==" ":
                        res.append(j)
                    else:
                        val_yes.append(j)
                val_yes = int("".join(map(str, val_yes)))
                val_no = int("".join(map(str, val_no)))

                if val_yes>val_no:
                    print(offset,'\033[1m',"YES")
                    print('\033[0m')
                elif val_no>val_yes:
                    print(offset,'\033[1m',"NO")
                    print('\033[0m')
                else:
                    print(offset,'\033[1m',"Tie")
                    print('\033[0m')

recurse(left, right, threshold, features, 0,0)