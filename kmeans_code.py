
def jaccarddistance(alltaskpaths):
    npaths = len(alltaskpaths)
    jd = numpy.ones((npaths, npaths))
    allpathsset = [set(path) for path in alltaskpaths]
    tmp = ((idx1, idx2, len(allpathsset[idx1].intersection(allpathsset[idx2]))/len(
        allpathsset[idx1].union(allpathsset[idx2]))) for idx1 in range(npaths-1) for idx2 in range(1, npaths))
    for idx1, idx2, v in tmp:
        jd[idx1, idx2] = jd[idx2, idx1] = v
    return jd


# t is jaccarddistance
def getOptimalClusters(t):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model.fit(t)
    Sum_of_squared_distances = []
    # K = range(1,len(model.labels_)) if len(model.labels_) < 30 else range(1,30)
    K = range(1, len(model.labels_))
    print("Total possible clusters...range", K)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(t)
        Sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, Sum_of_squared_distances,
                     curve='convex', direction='decreasing')
    print("kn.knee", kn.knee)
    return kn.knee


def addclusterIDtoData(df, changes):
    print("111 > addclusterIDtoData call is in progress... and its df shape is", df.shape)
    caseidcol = changes['Case Id']
    activitycol = changes['Activity']
    timestamp = changes['Start Time']
    df[timestamp] = df[timestamp].astype('datetime64[ns]')
    # print("df[timestamp] astype 'datetime64[ns]' completed...")
    df[activitycol] = df[activitycol].astype("category")
    df["duration"] = df.groupby(caseidcol)[timestamp].transform(
        lambda x: x.diff()/numpy.timedelta64(1, 'h')).shift(-1)
    # addclusterIDtoData(df)
    casestaskpath = df.groupby(caseidcol)[activitycol].agg(
        taskpath=lambda x: tuple(OrderedDict.fromkeys(x.cat.codes)))
    tmp = casestaskpath.value_counts().reset_index(name="numcases")
    casestaskpath.reset_index(inplace=True)
    t = 1-jaccarddistance(tmp.taskpath)
    print("160 >Working on to get Optimal clusters")
    n = getOptimalClusters(t)
    clustering = AgglomerativeClustering(
        n_clusters=n, metric='precomputed', linkage='complete').fit(t)
    tmp["clusterid"] = clustering.labels_
    print("165 > cluster Id calculation completed...")
    casestaskpath = pandas.merge(casestaskpath, tmp, on="taskpath")
    df = pandas.merge(df, casestaskpath, on=caseidcol)
    df.drop(["numcases", "taskpath"], axis=1, inplace=True)
    # file_name = UPLOAD_FOLDER+'/'+model_inst.filenames
    # f = file_name.split(".")
    # df.to_excel(file_name, index=False)  if f[-1][0]=="x" else\
    #    df.to_csv(file_name, index=False)
    # print("58 > cluster ID - Refer file saved as ", file_name)

    return df
