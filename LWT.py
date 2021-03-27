#Core Francisco Park implemented a wavelet package inspired from https://arxiv.org/abs/1203.1513

import torch
import torch.fft
import numpy as np

import scipy.interpolate as sintp
import scipy.ndimage as sim

def Pk(image,keepdim=False):
    assert not image.requires_grad,"No gradient support"
    dim=len(image.size())
    assert dim==2 or dim==3, "image should be 2D or 3D"
    N=image.size(0)
    image_k=torch.abs(torch.fft.fftn(image))**2
    k_arr=torch.fft.fftfreq(N,1/N)
    k_p_dims=torch.meshgrid(*(k_arr for _ in range(dim)))
    ksqs=torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
    k_abs=torch.sqrt(ksqs)
    pk_len=int(torch.max(k_abs)+0.5)+1
    ks=torch.zeros(pk_len)
    Pks=torch.zeros(pk_len)
    counts=torch.zeros(pk_len)
    for i in range(pk_len):
        where=(k_abs>=i-0.5)*(k_abs<i+0.5)
        ks[i]+=torch.sum(k_abs[where])
        Pks[i]+=torch.sum(image_k[where])
        counts[i]+=torch.sum(where)
    ks=ks/counts
    Pks=Pks/counts
    sort_ind=torch.argsort(ks)
    ks=ks[sort_ind]
    Pks=Pks[sort_ind]
    if not keepdim:
        return ks,Pks
    res=np.zeros_like(image)
    intp=sintp.interp1d(ks.cpu().detach().numpy(),Pks.cpu().detach().numpy())
    return ks,Pks,torch.tensor(intp(k_abs.cpu().detach().numpy()))

def Bke(image,ks,rings):
    assert not image.requires_grad,"No gradient support"
    dim=len(image.size())
    assert dim==2 or dim==3, "image should be 2D or 3D"
    assert len(ks)==len(rings), "One ring for k needed"
    N=image.size(0)
    image_k=torch.fft.fftn(image)
    k3s=[]
    Bkes=[]
    temp=[]
    for ring in rings:
        temp.append(torch.fft.ifftn(image_k*ring).real)
    #TODO: triangular inequality
    for i in range(len(ks)):
        for j in range(len(ks)):
            for k in range(len(ks)):
                k3s.append((ks[i],ks[j],ks[k]))
                Bkes.append(torch.sum(temp[i]*temp[j]*temp[k]))
    return k3s,torch.stack(Bkes)

def Bkn(rings):
    Bkns=[]
    temp=[]
    for ring in rings:
        temp.append(torch.fft.ifftn(ring).real)
    #TODO: triangular inequality
    for i in range(len(rings)):
        for j in range(len(rings)):
            for k in range(len(rings)):
                Bkns.append(torch.sum(temp[i]*temp[j]*temp[k]))
    return torch.stack(Bkns)

def make_healpix_wavelets(N,NR=8,nside=2,dtype=torch.float64,rough_mem_limit=4000):
    import healpy
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

def make_icosahedron_wavelets(N,NR=8,nside=1):
    assert False, "Not doing this"
    #from https://en.wikipedia.org/wiki/Regular_icosahedron#/media/File:Icosahedron-golden-rectangles.svg

def make_wavelets(N,NR=6,NT=8,dtype=torch.float64,add_z=False,NZ=6,return_bases=False):
    k_arr=torch.fft.fftfreq(N,1/N).to(dtype=dtype)
    kx,ky=torch.meshgrid(k_arr,k_arr)
    k_abs=torch.sqrt(kx**2+ky**2)
    k_theta=torch.atan2(ky,kx)
    rs=torch.logspace(1,np.log2(N//2),NR+1,base=2)
    rs=torch.cat([torch.zeros(1),rs],dim=0).to(dtype=dtype)
    r_dists=(rs[1:]-rs[:-1])
    thetas=torch.arange(-1,NT+1).to(dtype=dtype)/NT*(2*np.pi if add_z else np.pi)
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
    wavelets=(radials[:,None]*angulars[None,:]).reshape(-1,N,N)
    dcw=(k_abs<=rs[1]).to(dtype=dtype)
    dcw*=(1-torch.sum(wavelets,dim=0))
    wavelets=torch.cat([dcw[None,:,:],wavelets],dim=0).to(dtype=dtype)
    if add_z:
        _,_,kz=torch.meshgrid(k_arr,k_arr,k_arr)
        rs=torch.logspace(1,np.log2(N//2),NZ+1,base=2)
        rs=torch.cat([torch.zeros(1),rs],dim=0).to(dtype=dtype)
        r_dists=(rs[1:]-rs[:-1])
        zs=[]
        for r,prev_dist,post_dist in zip(rs[1:-1],r_dists[:-1],r_dists[1:]):
            diff=kz-r
            t=torch.abs(diff)
            t_prev=t/prev_dist
            t_post=t/post_dist
            prev=(2*t_prev**3-3*t_prev**2+1)*(t<prev_dist)*(diff<=0)
            post=(2*t_post**3-3*t_post**2+1)*(t<post_dist)*(diff>0)
            zs.append(post+prev)
        zs=torch.stack(zs)
        print("zs shape",zs.shape)
        wavelets=(zs[None,:,:,:,:]*wavelets[:,None,:,:,None]).reshape(-1,N,N,N)
        if return_bases:
            return wavelets,{"r":radials,"theta":angulars,"z":zs}
        return wavelets
    if return_bases:
        return wavelets,{"r":radials,"theta":angulars}
    return wavelets

def make_rings(N,NR=20,dtype=torch.float64,btype="log",ret_ks=False):
    k_arr=torch.fft.fftfreq(N,1/N).to(dtype=dtype)
    kx,ky=torch.meshgrid(k_arr,k_arr)
    k_abs=torch.sqrt(kx**2+ky**2)
    k_theta=torch.atan2(ky,kx)
    if btype=="log":
        rs=torch.logspace(1,np.log2(N//2),NR+1,base=2)
    elif btype=="lin":
        rs=torch.linspace(2,N//2,NR+1)
    else:
        assert False, "Not implemented"
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
    radials=torch.stack(radials)

    dcw=(k_abs<=rs[1]).to(dtype=dtype)
    dcw*=(1-torch.sum(radials,dim=0))

    rings=torch.cat([dcw[None,:,:],radials],dim=0).to(dtype=dtype)
    if ret_ks:
        return rings,torch.cat((torch.tensor([0.],dtype=dtype),rs[1:-1]),)
    return rings

def LWT_R(image,wavelets,m=2,rsnl="abs",verbose=False,pknorm=False):
    dim=len(image.size())
    assert dim==2 or dim==3, "image should be 2D or 3D"
    N=image.size(0)
    assert m in [0,1,2], "m should be 0,1,2"
    assert all([s==N for s in image.size()]),"image not "+str(dim)+"-cube"
    assert len(wavelets[0].size())==dim,"Wavelets not right."

    assert rsnl in ["abs","abs2","logabsp1"],"Only abs,abs2,logabsp1 implemented"
    if rsnl=="abs":
        rsnl=torch.abs
    elif rsnl=="abs2": #Yes, I know parseval's theorem, but didn't care yet....
        rsnl=lambda x: torch.abs(x)**2
    elif rsnl=="logabsp1":
        rsnl=lambda x: torch.log(torch.abs(x)+1)
    else:
        assert False,"bug cfpark00"

    if pknorm:
        ks,Pks,pknd=Pk(image,keepdim=True)

    Nw=len(wavelets)
    coeffs=[]
    std,mean=torch.std_mean(image)
    coeffs.append(mean)
    coeffs.append(std)
    
    image=(image-m)/(std+1e-8)

    
    image_k=torch.fft.fftn(image)
    if pknorm:
        image_k=image_k/(pknd+1e-9)
        coeffs.extend(Pks)
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

def wavelet_to_mms_vals(wavelets):
    dim=len(wavelets.shape)-1
    assert dim==2 or dim==3,"2 or 3 dim"
    wavelets_shifted=torch.fft.fftshift(wavelets)
    mms=[]
    vals=[]
    logicals=wavelets>1e-13
    for wavelet,logical in zip(wavelets_shifted,logicals):
        ms=[]
        Ms=[]
        for i in range(dim):
            inds=np.nonzero(logical.sum(i))[0]
            ms.append(np.min(inds))
            Ms.append(np.max(inds))
        mms.append(np.array([ms,Ms]))
        if dim==3:
            vals.append(wavelet[ms[0]:Ms[0],ms[1]:Ms[1],ms[2]:Ms[2]]
        elif dim==2:
            vals.append(wavelet[ms[0]:Ms[0],ms[1]:Ms[1]]

def LWT_R_abs2_fast(image,wavelet_mms,wavelet_vals,m=2,verbose=False,pknorm=False):
    dim=len(image.size())
    N=image.size(0)

    if pknorm:
        ks,Pks,pknd=Pk(image,keepdim=True)

    Nw=len(wavelet_mms)
    assert Nw==len(wavelet_vals)
    for wavelet in wavelets:
        waveletsmms
    coeffs=[]
    std,mean=torch.std_mean(image)
    coeffs.append(mean)
    coeffs.append(std)
    
    image=(image-m)/(std+1e-8)

    
    image_k=torch.fft.fftshift(torch.fft.fftn(image))
    if pknorm:
        image_k=image_k/(pknd+1e-9)
        coeffs.extend(Pks)
    if m==2:
        coeffs2=[]
    if m==0:
        return torch.stack(coeffs)
    for w1 in range(Nw):
        if verbose:
            print("Wavelet:",str(w1+1),"/",str(Nw))

        im1_r=torch.fft.ifftn(image_k*wavelets[w1])
        im1_r=im1_r.real**2+im1_r.imag**2
        coeffs.append(torch.sum(im1_r))
        if m==1:
            continue
        im1_f=torch.fft.fftn(torch.sqrt(im1_r))
        for w2 in range(Nw):
            im2_f=im1_f*wavelets[w2]
            im2_f=im2_f.real**2+im2_f.imag**2
            coeffs2.append(torch.sum(im2_f))
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


def LIest(image,scales=None,NR=8,max_offset=10):
    N=image.size(0)
    if scales is None:
        scales=torch.logspace(0,np.log2(N//2),NR,base=2).numpy()
    coeffs=[]
    for i in range(len(scales)+1):
        if i==0:
            im=image
        else:
            im=torch.tensor(sim.gaussian_filter(image.numpy(),scales[i-1],mode="wrap"))
        for offset in range(1,max_offset):
            coeffs.append(torch.sum(torch.sqrt(torch.abs(torch.roll(im,offset,0)*im)))/(N*N)  )
            coeffs.append(torch.sum(torch.sqrt(torch.abs(torch.roll(im,offset,1)*im)))/(N*N)  )
    return torch.stack(coeffs)

def disp(wst,nf,m=2,norm=True,flip_rl=False,r=8,l=8):
    if type(wst)!=np.ndarray:
        wst=wst.detach().numpy()
    assert m==2,"only for m=2 for now"
    assert len(wst)==2+nf+nf**2,"length mismatch"
    feed=wst.copy()
    if flip_rl:
        feed[3:2+nf]=feed[3:2+nf].reshape(l,r).T.reshape(-1)
        
        temp=feed[2+nf:].reshape(1+r*l,1+r*l)
        temp[1:,0]=temp[1:,0].reshape(l,r).T.reshape(-1)
        temp[0,1:]=temp[0,1:].reshape(l,r).T.reshape(-1)
        temp[1:,1:]=temp[1:,1:].reshape(l,r,l,r).transpose(1,0,3,2).reshape(r*l,r*l)
        feed[2+nf:]=temp.reshape(-1)

    if norm:
        def normalize(vals):
            return (vals-np.mean(vals))/np.std(vals)
    else:
        normalize= lambda x:x
    im=np.zeros((nf,3+nf))
    im[:,0]=feed[0] if not norm else 0
    im[:,1]=feed[1] if not norm else 0
    im[:,2]=normalize(feed[2:2+nf])
    im[:,3:]=normalize(feed[2+nf:].reshape(nf,nf))
    return im

def asinhnorm(im):
    std=np.std(im)
    return np.arcsinh(im/std)
