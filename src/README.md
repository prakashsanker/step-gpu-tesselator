What do I need to do for this checkpoint?


I have to create a function isEar, which will tell me if a given vertex is an ear of a polygon. 


1. First I need to check to see if the points are ordered in CCW.


    How do we write this code?
        Initialize data
            a. Points buffer --> these are all of the actual vertex points. This looks like this [x0, y0, z1, 0, x1, y1, z1, 0]  --> with padding. 
            b. Output Indices Buffer --> this holds the indexes for each triangle. 
            c. Vertex Is Ear Buffer --> this holds whether that vertex is an ear or not
            d. Previous Vertex Buffer --> for vertex i, this holds the previous vertex index (the index in the points buffer indicating x0, x1, x2)
            e. Next Vertex buffer  --> for vertex i, this holds the next vertex index (the index in the points buffer indicating x0, x1, x2)
            f. activeCount --> this holds the number of active vertices. 
        b. Run the convexity-check for each vertex --> this now gives you an array of classifications. This is stored in memory. And is your first list of convex. 
        c. Now you have to do isEar
            For each vertex run the shader -->
                a. If not convex, return
                b. If convex,
                    Find triangle.
                    Go through other points. 
                    See if this vertex is an ear.
                c. Now..at the same time, I know whether a number of vertices are ears or not.
        d. Now I have to extract the triangle. So this is another compute pipeline. It runs once. 
            i. Here I pick the lowest indexed ear. 
            ii. I take the previous vertex in the vertices array, I take the next vertex in the vertices array. 
            iii. I add all thee vertices to the indices storage. I also maintain the starting index for the next triangle in the indices. 
                By add all three vertices, I mean add the vertex index. 
            iv. I also need to maintain a vertex list of whether each vertex has been processed or not. 
            v. I set this vertex to be inactive. 
        e. Now I need to repeat the process from step b. 