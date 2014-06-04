subroutine applyk(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, dimb, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim, dimb
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dimb)
  real(8) :: beta(num_nodes_y, dimb)
  real(8) :: Kh

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(num_nodes_y, dimb) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dimb) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dimb)
  real(8) :: c_(5, 5)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

  !$omp parallel do private(k,l,ut,lpt,Kh,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh = 1.0
           else
              Kh = exp(-0.5*ut)
           end if
        else           
           if (ut < 1e-8) then
              Kh = 1.0
           else
              lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
              Kh = lpt * exp(-1.0*ut)
           end if
        end if
        df = df + Kh * beta(l,:)
     end do
     f(k,:) = df
  end do
  !$omp end parallel do
end subroutine applyK

subroutine applyk_and_diff(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, f, f2)
  implicit none
  integer :: num_nodes, num_nodes_y, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes)
  real(8) :: f2(num_nodes,dim)
  real(8) :: beta(num_nodes_y)

  real(8) :: Kh, Kh_diff

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(num_nodes_y) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes) :: f
  !f2py real(8), intent(out), dimension(num_nodes,dim) :: f2

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df, df2(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh,Kh_diff,df,df2) shared &
  !$omp& (num_nodes, num_nodes_y, f, f2, sig, ord, beta, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     df2 = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh = 1.0
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh = exp(-0.5*ut)
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh = 1.0
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
              Kh = lpt * exp(-1.0*ut)
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df + Kh * beta(l)
        df2 = df2 + Kh_diff * 2*(x(k,:)-y(l,:))* beta(l)
     end do
     f(k) = f(k) + df
     f2(k,:) = f2(k,:) + df2
  end do
  !$omp end parallel do
end subroutine applyK_and_Diff
