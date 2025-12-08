export function isCounterClockWise(points): boolean{
    /*
        We want to calculate the signed area here.  If the area is positive, then we are CCW. Else negative and we have to reverse the array. 

        We need to do 1/2 * 
    */
    let sum = 0;
    console.log("POINTS", points);
    for (let i = 0; i < points.length; i++) {
        const currentPoint = points[i];
        let nextPoint;
        if (i === points.length -1 ) {
            nextPoint = points[0];
        } else {
            nextPoint = points[i+1];
        }
        const x1 = currentPoint[0];
        const y1 = currentPoint[1];
        const x2 = nextPoint[0];
        const y2 = nextPoint[1];
        console.log("X1", x1);
        console.log("Y1", y1);
        console.log("X2", x2);
        console.log("Y2", y2);
        const intermediateSum = x1*y2 - x2*y1;
        console.log("INTERMEDIATE SUM", intermediateSum);
        console.log("-=======");
        sum+= intermediateSum;
        
    }
    console.log("SUM", sum);
    if (sum === 0) {
        throw new Error("collinear or degenerate polygon");
    }

    if (sum > 0) {
        return true;
    }
    return false;
}