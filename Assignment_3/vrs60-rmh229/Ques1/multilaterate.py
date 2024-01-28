import sys
import numpy as np

def check_soln(points, target, distance):
    if np.isnan(target).any():
        return False
    for i in range(0,(int)(points.size/3)):
        if np.linalg.norm(points[i] - target) != distance[i]:
            return False
    return True

def multilaterate(distances):

    point1 = np.array(distances[0][:3])
    point2 = np.array(distances[1][:3])
    point3 = np.array(distances[2][:3])
    point4 = np.array(distances[3][:3])

    dist1 = distances[0][-1]
    dist2 = distances[1][-1]
    dist3 = distances[2][-1]
    dist4 = distances[3][-1]

    # Eliminate duplicate points

    points = np.array([point1, point2, point3, point4])
    distance = np.array([dist1, dist2, dist3, dist4])
    np.sort(points, axis=None)
    if (points[2]==points[3]).all():
        if distance[2] == distance[3]:
            points = np.delete(points, 3, 0)
            distance = np.delete(distance, 3)
        else:
            raise ValueError('Invalid Input, points are same but distances are not')
        
        
    if (points[1]==points[2]).all():
        if distance[1] == distance[2]:
            points = np.delete(points, 2, 0)
            distance = np.delete(distance, 2)
        else:
            raise ValueError('Invalid Input, points are same but distances are not')
        
        
    if (points[0]==points[1]).all():  
        if distance[0] == distance[1]:
            points = np.delete(points, 1, 0)
            distance = np.delete(distance, 1)
        else:
            raise ValueError('Invalid Input, points are same but distances are not')

  
    if points.shape[0] == 4:
        eX = (point2 - point1)/np.linalg.norm(point2 - point1)
        i = np.dot(eX, (point3 - point1))
        eY = (point3 - point1 - (i * eX))/(np.linalg.norm(point3 - point1 - (i * eX)))
        eZ = np.cross(eX, eY)
        d0 = np.linalg.norm(point2 - point1)
        j = np.dot(eY, (point3 - point1))
        x = ((dist1 ** 2) - (dist2 ** 2) + (d0 ** 2))/(2 * d0)
        y = (((dist1 ** 2) - (dist3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i/j) * (x))
        # Assume both positive and negative values of the point in z-axis
        z1 = np.sqrt(dist1 ** 2 - x ** 2 - y ** 2)
        z2 = np.sqrt(dist1 ** 2 - x ** 2 - y ** 2) * (-1)
        answedist1 = point1 + (x * eX) + (y * eY) + (z1 * eZ)
        answedist2 = point1 + (x * eX) + (y * eY) + (z2 * eZ)
        d1 = np.linalg.norm(point4 - answedist1)
        d2 = np.linalg.norm(point4 - answedist2)
        # Check which of the following is solution gives a closer estimate of distance form 4th point
        if np.abs(dist4 - d1) < np.abs(dist4 - d2):
            ans = round(answedist1[0], 6), round(answedist1[1], 6), round(answedist1[2], 6)
            if np.isnan(ans).any():
                raise ValueError('Not Solvable')
            else:
                return ans
        else:
            ans = round(answedist2[0], 6), round(answedist2[1], 6), round(answedist2[2], 6)
            if np.isnan(ans).any():
                raise ValueError('Not Solvable')
            else:
                return ans

    elif points.shape[0] == 3:
        point1 = points[0]
        point2 = points[1]
        point3 = points[2]

        eX = (point2 - point1)/np.linalg.norm(point2 - point1)
        i = np.dot(eX, (point3 - point1))
        eY = (point3 - point1 - (i * eX))/(np.linalg.norm(point3 - point1 - (i * eX)))
        eZ = np.cross(eX, eY)
        d0 = np.linalg.norm(point2 - point1)
        j = np.dot(eY, (point3 - point1))
        x = ((dist1 ** 2) - (dist2 ** 2) + (d0 ** 2))/(2 * d0)
        y = (((dist1 ** 2) - (dist3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i/j) * (x))
        # Assume both positive and negative values of the point in z-axis
        z1 = np.sqrt(dist1 ** 2 - x ** 2 - y ** 2)
        z2 = np.sqrt(dist1 ** 2 - x ** 2 - y ** 2) * (-1)
        answedist1 = point1 + (x * eX) + (y * eY) + (z1 * eZ)
        answedist2 = point1 + (x * eX) + (y * eY) + (z2 * eZ)
        ans = round(answedist1[0], 6), round(answedist1[1], 6), round(answedist1[2], 6)
        ans2 = round(answedist2[0], 6), round(answedist2[1], 6), round(answedist2[2], 6)

        # To check which of the 2 possible points satisfy all the constraints

        if check_soln(points, ans, distance):
            return ans
        if check_soln(points, ans2, distance):
            return ans2
        raise ValueError('Not Solvable')

    elif points.shape[0] == 2:
        # To check if the distance between both the points is same as sum of their radii
        if np.linalg.norm(points[0] - points[1]) == (distance[0] + distance[1]):
            vector = np.multiply((points[0] - points[1]), distance[0]/(distance[0]+distance[1]))
            ans = list(points[1]+vector)
            ans = [round(elem, 6) for elem in ans]
            return ans
        else:
            raise ValueError('Cannot solve, only two unique point provided')

    elif points.shape[0] == 1:
        # If only one point is unique, then check if the distance is zero, ie the point is the solution
        if dist1 == 0:
            return list(point1)
        else:
            raise ValueError('Cannot solve, only one unique point provided')
    else:
        raise ValueError('Not Solvable')

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) == 1):
        print("Please enter data file name.")
        exit()
    
    filename = sys.argv[1]

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    distances = []
    for line in range(0, len(lines)):
        distances.append(list(map(float, lines[line].split(' '))))

    # Print out the data
    print ("The input four points and distances, in the format of [x, y, z, d], are:")
    for p in range(0, len(distances)):
        print (*distances[p]) 

    # Call the function and compute the location 
    location = multilaterate(distances)
    print ()
    print ("The location of the point is: " + str(location))
