import torch

def modify_threshold(A, threshold):
    # A = torch.matmul(x.transpose(1, 2), y)
    zero = torch.zeros_like(A)
    one = torch.ones_like(A)
    A = torch.where(A < threshold, zero, one) # two sides
    # A = torch.where(A > threshold, one, A) # one side
    return A

def modify_topk(x, y, top_k=3, num=1):
	A = torch.rand(x.shape[0], y.shape[0])

	for i in range(x.shape[0]):
		for j in range(y.shape[0]):
			A[i][j] = torch.abs(torch.sum(x[i] - y[j]))

	temp = torch.ones_like(A) * 100
	A = temp - A

	_, idx = torch.topk(A, top_k, dim=1, largest=True)
	for i in range(idx.shape[0]):
		for index in idx[i]:
			A[i][index] = num

	zero = torch.zeros_like(A)
	A = torch.where(A == num, A, zero)
	return A

def modify_dist(x, y, top_k=3, num=1):
	A = torch.rand(x.shape[0], y.shape[0])

	for i in range(x.shape[0]):
			# A[i][j] = torch.cosine_similarity(x[i], y[j], dim=-1)
			A[i] = torch.pairwise_distance(x[i].unsqueeze(0), y, p=2)

	print(A)
	B = torch.pairwise_distance(x[0].unsqueeze(0), y, p=2)
	print(B)
	exit()
	temp = torch.ones_like(A) * 100
	A = temp - A

	_, idx = torch.topk(A, top_k, dim=1, largest=True)
	for i in range(idx.shape[0]):
		for index in idx[i]:
			A[i][index] = num

	zero = torch.zeros_like(A)
	A = torch.where(A == num, A, zero)
	return A

if __name__ == '__main__':
	x = torch.rand(5, 3) * 10
	y = torch.tensor([[1., 1., 1.],
					  [2., 2., 2.]])
	print(x)
	# print(modify_threshold(x, 0.5))
	# print(modify_topk(x, x, 3))
	print(modify_dist(x, x))