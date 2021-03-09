import torch
import torch.fft
import numpy as np
import healpy

def make_healpix_wavelets(N,NR=8,nside=2,dtype=torch.float64,rough_mem_limit=4000):
    assert dtype==torch.float64 or dtype==torch.float32, "only torch.float64 or torch.float32 allowed"
    mem=NR*12*(nside**2)*(N**3)*(8 if dtype==torch.float64 else 4)/1e6
    assert mem<rough_mem_limit, "This probably uses "+str(mem)+" MB"
    k_arr=torch.fft.fftfreq(N,1/N).to(dtype=dtype)
    kx,ky,kz=torch.meshgrid(k_arr,k_arr,k_arr)
    k_abs=torch.sqrt(kx**2+ky**2+kz**2)
    k_phi=torch.atan2(ky,kx)
    k_theta=torch.atan2(torch.sqrt(kx**2+ky**2),kz)

    rs=torch.logspace(1,np.log2(N//2),NR+1,base=2)
    rs=torch.cat([torch.zeros(1),rs],dim=0).to(dtype=dtype)
    r_dists=(rs[1:]-rs[:-1])
    radials=[]
    for r,prev_dist,post_dist in zip(rs[1:-1],r_dists[:-1],r_dists[1:]):
        diff=k_abs-r
        t=torch.abs(diff)
        t_prev=t/prev_dist
        t_post=t/post_dist
        prev=(2*t_prev**3-3*t_prev**2+1)*(t<prev_dist)*(diff<=0)
        post=(2*t_post**3-3*t_post**2+1)*(t<post_dist)*(diff>0)
        radials.append(post+prev)
    radials=torch.stack(radials).to(dtype=dtype)

    inds,weights=healpy.pixelfunc.get_interp_weights(nside,k_theta.numpy(),k_phi.numpy())
    indarr=np.arange(N)
    indarrx,indarry,indarrz=np.meshgrid(indarr,indarr,indarr,indexing="ij")

    angulars=torch.zeros(healpy.pixelfunc.nside2npix(nside),N,N,N,dtype=dtype)#npix is 12*nside**2
    angulars[inds[0],indarrx,indarry,indarrz]+=torch.tensor(weights[0],dtype=dtype)
    angulars[inds[1],indarrx,indarry,indarrz]+=torch.tensor(weights[1],dtype=dtype)
    angulars[inds[2],indarrx,indarry,indarrz]+=torch.tensor(weights[2],dtype=dtype)
    angulars[inds[3],indarrx,indarry,indarrz]+=torch.tensor(weights[3],dtype=dtype)

    #make the DC part.. yes this is very inefficient.
    wavelets=(radials[None,:]*angulars[:,None]).reshape(-1,N,N,N)
    dcw=(k_abs<=rs[1]).to(dtype=dtype)
    dcw*=(1-torch.sum(wavelets,dim=0))

    angulars=angulars[:healpy.pixelfunc.nside2npix(nside)//2]
    wavelets=(radials[None,:]*angulars[:,None]).reshape(-1,N,N,N)
    wavelets=torch.cat([dcw[None,...],wavelets],dim=0).to(dtype=dtype)
    return wavelets


def make_wavelets(N,NR=8,NT=8,dtype=torch.float64):
    k_arr=torch.fft.fftfreq(N,1/N).to(dtype=dtype)
    kx,ky=torch.meshgrid(k_arr,k_arr)
    k_abs=torch.sqrt(kx**2+ky**2)
    k_theta=torch.atan2(ky,kx)
    rs=torch.logspace(1,np.log2(N//2),NR+1,base=2)
    rs=torch.cat([torch.zeros(1),rs],dim=0).to(dtype=dtype)
    r_dists=(rs[1:]-rs[:-1])
    thetas=torch.arange(-1,NT+1).to(dtype=dtype)/NT*np.pi
    t_dists=(thetas[1:]-thetas[:-1])
    radials=[]
    for r,prev_dist,post_dist in zip(rs[1:-1],r_dists[:-1],r_dists[1:]):
        diff=k_abs-r
        t=torch.abs(diff)
        t_prev=t/prev_dist
        t_post=t/post_dist
        prev=(2*t_prev**3-3*t_prev**2+1)*(t<prev_dist)*(diff<=0)
        post=(2*t_post**3-3*t_post**2+1)*(t<post_dist)*(diff>0)
        radials.append(post+prev)
    radials=torch.stack(radials)
    angulars=[]
    for t,prev_dist,post_dist in zip(thetas[1:-1],t_dists[:-1],t_dists[1:]):
        diff=k_theta-t
        t=torch.abs(diff)
        t_prev=t/prev_dist
        t_post=t/post_dist
        prev=(2*t_prev**3-3*t_prev**2+1)*(t<prev_dist)*(diff<=0)
        post=(2*t_post**3-3*t_post**2+1)*(t<post_dist)*(diff>0)
        angulars.append(post+prev)
    angulars=torch.stack(angulars)
    wavelets=(radials[None,:]*angulars[:,None]).reshape(-1,N,N)
    dcw=torch.zeros(1,N,N)
    dcw[0,0,0]=1
    wavelets=torch.cat([dcw,wavelets],dim=0).to(dtype=dtype)
    return wavelets


def LWT_R(image,wavelets,m=2,rsnl="abs",verbose=False):
    dim=len(image.size())
    assert dim==2 or dim==3, "image should be 2D or 3D"
    N=image.size(0)
    assert m in [0,1,2], "m should be 0,1,2"
    assert all([s==N for s in image.size()]),"image not "+str(dim)+"-cube"
    assert len(wavelets.size())==dim+1,"Wavelets not right."
    assert all([s==N for s in wavelets.size()[1:]   ]),"Wavelets not right."

    assert rsnl in ["abs","abs2","logabsp1"],"Only abs,abs2,logabsp1 implemented"
    if rsnl=="abs":
        rsnl=torch.abs
    elif rsnl=="abs2": #Yes, I know parseval's theorem, but didn't care yet....
        rsnl=lambda x: torch.abs(x)**2
    elif rsnl=="logabsp1":
        rsnl=lambda x: torch.log(torch.abs(x)+1)
    else:
        assert False,"bug cfpark00"

    Nw=wavelets.size(0)
    coeffs=[]
    std,mean=torch.std_mean(image)
    coeffs.append(mean)
    coeffs.append(std)
    
    image=(image-m)/(std+1e-8)
    
    image_k=torch.fft.fftn(image)
    if m==2:
        coeffs2=[]
    if m==0:
        return torch.stack(coeffs)
    for w1 in range(Nw):
        if verbose:
            print("Wavelet:",str(w1+1),"/",str(Nw))
        im1_r=rsnl(torch.fft.ifftn(image_k*wavelets[w1]))
        coeffs.append(torch.sum(im1_r))
        if m==1:
            continue
        im1_f=torch.fft.fftn(im1_r)
        for w2 in range(Nw):
            im2_r=rsnl(torch.fft.ifftn(im1_f*wavelets[w2]))
            coeffs2.append(torch.sum(im2_r))
    if m==2:
        coeffs.extend(coeffs2)
    return torch.stack(coeffs)


def LWT2D_F(image,wavelets,m=2):
    assert False, "Not doing this"
    assert m<=2, "m max currently 2"
    wavelets=wavelets/torch.sum(wavelets,axis=(1,2))[:,None,None]
    print("Currently just filter response")
    N=image.size(0)
    Nw=wavelets.size(0)
    coeffs=[]
    std,mean=torch.std_mean(image)
    coeffs.append(mean)
    coeffs.append(std)
    
    image=(image-m)/(std+1e-8)
    
    image_k=torch.fft.fftn(image)
    limage_k=torch.log(torch.abs(image_k)+1e-8).reshape(-1)
    resps=torch.matmul(wavelets.reshape(-1,N*N),lx_k)
    coeffs.extend(resps)
    return torch.stack(coeffs)