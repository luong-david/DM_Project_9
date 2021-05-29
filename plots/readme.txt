dot files are generated to visualize decision trees in the classification analysis.

if Graphviz and Pydot are installed, uncomment the following code in the plotDecisionTree() function in functions.py.

    # Convert to png using system command (requires Graphviz)
    #from subprocess import call
    #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    
    #import pydot (requires pydot)
    #(graph,) = pydot.graph_from_dot_file('classifier.dot')
    #graph.write_png('tree.png')

else, can use a free online service to convert dot files to png files e.g. https://onlineconvertfree.com/