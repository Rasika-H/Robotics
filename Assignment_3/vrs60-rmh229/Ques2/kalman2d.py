import sys
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[0.0002, 0.00004], [0.00003, 0.0001]])
R = np.array([[0.0100, 0.0040], [0.0030, 0.0200]])

#predict step
def predict(x, u, p):
    x_intermed = np.add(x, u)
    p_intermed = np.add(p, Q)

    return x_intermed, p_intermed

#update step
def update(p_intermed, x_intermed, v):
    k_filter = np.divide(p_intermed, np.add(p_intermed, R))
    x_estimate = np.add(x_intermed, np.matmul(k_filter, np.subtract(v, x_intermed)))
    p_estimate = np.multiply(np.subtract(np.eye(2), k_filter), p_intermed)

    return x_estimate, p_estimate


if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 5):
        print ("Four arguments required: python kalman2d.py [datafile] [x1] [x2] [lambda]")
        exit()
    
    filename = sys.argv[1]
    x10 = float(sys.argv[2])
    x20 = float(sys.argv[3])
    scaler = float(sys.argv[4])

    #Position and probability matrix update
    x = np.array([[x10], [x20]])
    p = np.array([[scaler, 0], [0, scaler]])

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    data = []
    for line in range(0, len(lines)):
        data.append(list(map(float, lines[line].split(' '))))

    # Print out the data
    print ("The input data points in the format of 'k [u1, u2, z1, z2]', are:")
    resultX1 = []
    resultX2 = []
    resultV1 = []
    resultV2 = []

    #iterating over all data
    for it in range(0, len(data)):
        u1 = data[it][0]
        u2 = data[it][1]
        v1 = data[it][2]
        v2 = data[it][3]

        print("1\n")

        u = np.array([[u1], [u2]])
        v = np.array([[v1], [v2]])

        print("2\n")
        # get the U and V
        x_intermed, p_intermed = predict(x, u, p)
        x, p = update(p_intermed, x_intermed, v)

        print("3\n")
        resultX1.append(x[0])
        resultX2.append(x[1])
        resultV1.append(v[0])
        resultV2.append(v[1])

        print("X : ")
        print(x)
        print("V : ")
        print(v)
        print("\n")

    #plotting all data
    plt.plot(resultX1, resultX2, '-o', label='Estimate')
    plt.plot(resultV1, resultV2, '-ro', label='Measurement')
    plt.xlabel("x1/v1")
    plt.ylabel("x2/v2")
    plt.title("Estimation vs Measurement")
    plt.legend()
    plt.show()

