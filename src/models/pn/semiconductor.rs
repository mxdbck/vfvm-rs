use super::pn::PnJunctionParams;
use crate::discretization::mesh::{Cell, Face};
use crate::physics::bc::Field;
use crate::physics::functional::FunctionalPhysics;
use num_dual::{DualDVec64, DualNum};

/// Creates a FunctionalPhysics object configured for the drift-diffusion model.
pub fn setup_semiconductor_physics(
    params: PnJunctionParams,
) -> FunctionalPhysics<DualDVec64, PnJunctionParams> {
    let reaction = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], cell: &Cell, data: &PnJunctionParams| {
            let (psi, phi_n, phi_p) = (&u[0], &u[1], &u[2]);

            let ni = DualDVec64::from_re(data.ni);
            let tau_p = data.tau_p;
            let tau_n = data.tau_n;
            let c_dop = DualDVec64::from_re(data.c_doping[cell.id]);

            let n = (psi - phi_n).exp() * data.ni;
            let p = (phi_p - psi).exp() * data.ni;

            // Poisson source
            f[0] = -(&p - &n + c_dop);

            // SRH recombination (placeholder form)
            let numerator = &n * &p - &ni * &ni;
            let denom = (n + &ni) * tau_p + (p + &ni) * tau_n;
            let srh = numerator / denom;

            f[1] = -&srh;
            f[2] = -srh;
        },
    );

    // FLUX (simplified central differences placeholder)
    let flux = Box::new(
        |f: &mut [DualDVec64],
         u_k: &[DualDVec64],
         u_l: &[DualDVec64],
         _face: &Face,
         data: &PnJunctionParams| {
            let (psi_k, phi_n_k, phi_p_k) = (&u_k[0], &u_k[1], &u_k[2]);
            let (psi_l, phi_n_l, phi_p_l) = (&u_l[0], &u_l[1], &u_l[2]);

            // Poisson flux (-psi gradient)
            f[0] = psi_k - psi_l;

            // Electron flux (ad-hoc averaging)
            let n_k = (psi_k - phi_n_k).exp() * data.ni;
            let n_l = (psi_l - phi_n_l).exp() * data.ni;
            let n_avg = (&n_k + &n_l) * 0.5;
            f[1] = n_avg * (phi_n_l - phi_n_k) * (-data.dn);

            // Hole flux
            let p_k = (phi_p_k - psi_k).exp() * data.ni;
            let p_l = (phi_p_l - psi_l).exp() * data.ni;
            let p_avg = (&p_k + &p_l) * 0.5;
            f[2] = p_avg * (phi_p_l - phi_p_k) * (data.dp);
        },
    );

    let storage = Box::new(
        |f: &mut [DualDVec64], u: &[DualDVec64], _cell: &Cell, data: &PnJunctionParams| {
            let (psi, phi_n, phi_p) = (&u[0], &u[1], &u[2]);
            f[0] = DualDVec64::from_re(0.0); // No time derivative for Poisson
            f[1] = (psi - phi_n).exp() * DualDVec64::from_re(data.ni);
            f[2] = (phi_p - psi).exp() * DualDVec64::from_re(data.ni);
        },
    );

    let fields = vec![
        Field::from("psi"),
        Field::from("phi_n"),
        Field::from("phi_p"),
    ];
    FunctionalPhysics::new(fields, params, flux, reaction, storage)
}
