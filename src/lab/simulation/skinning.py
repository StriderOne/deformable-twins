"""GPU equivalent of the reference local rigid-fit Gaussian binding."""
import numpy as np
import warp as wp


@wp.kernel
def fit_rotations(q: wp.array[wp.vec3], adj: wp.array2d[int], rest: wp.array2d[wp.vec3],
                  rotations: wp.array[wp.mat33]):
    i=wp.tid()
    covariance=wp.mat33(0.)
    for k in range(adj.shape[1]):
        covariance += wp.outer(q[adj[i,k]]-q[i],rest[i,k])
    u,s,v=wp.svd3(covariance)
    sign=wp.where(wp.determinant(u*wp.transpose(v))<0.,-1.,1.)
    rotations[i]=u*wp.diag(wp.vec3(1.,1.,sign))*wp.transpose(v)


@wp.kernel
def deform_centers(q: wp.array[wp.vec3], rotations: wp.array[wp.mat33], ids: wp.array2d[int],
                   offset: wp.array2d[wp.vec3], weights: wp.array2d[float], output: wp.array[wp.vec3]):
    i=wp.tid()
    result=wp.vec3(0.)
    for k in range(ids.shape[1]):
        j=ids[i,k]
        result += weights[i,k]*(rotations[j]*offset[i,k]+q[j])
    output[i]=result


class GpuGaussianBinding:
    def __init__(self, binding, device):
        self.device=wp.get_device(device)
        self.adj=wp.array(binding.adj.astype(np.int32),dtype=int,device=device)
        self.rest=wp.array(binding.adj_rest,dtype=wp.vec3,device=device)
        self.ids=wp.array(binding.ids.astype(np.int32),dtype=int,device=device)
        self.offset=wp.array(binding.offset,dtype=wp.vec3,device=device)
        self.weights=wp.array(binding.weights.astype(np.float32),dtype=float,device=device)
        self.rotations=wp.empty(len(binding.rest),dtype=wp.mat33,device=device)
        self.output=wp.array(binding.centers,dtype=wp.vec3,device=device)

    def deform(self, particles):
        if particles.shape!=(self.adj.shape[0],) or particles.device!=self.device:
            raise ValueError('Particle shape/device mismatch for Gaussian binding')
        wp.launch(fit_rotations,self.adj.shape[0],[particles,self.adj,self.rest,self.rotations],device=self.device)
        wp.launch(deform_centers,self.ids.shape[0],
                  [particles,self.rotations,self.ids,self.offset,self.weights,self.output],device=self.device)
        return self.output
