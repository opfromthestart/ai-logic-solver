use std::{
    rc::Rc,
    sync::{Arc, RwLock},
};

use coaster::{frameworks::{cuda::get_cuda_backend, native::get_native_backend}, SharedTensor};
use juice::{
    layer::{LayerConfig, LayerType},
    layers::{LinearConfig, SequentialConfig},
    solver::{Solver, SolverConfig},
    util::write_batch_sample,
};

fn main() {
    let back = Rc::new(get_native_backend());

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[4, 2]);
    net_cfg.add_layer(LayerConfig::new(
        "1",
        LayerType::Linear(LinearConfig { output_size: 2 }),
    ));
    net_cfg.add_layer(LayerConfig::new("1s", LayerType::ReLU));
    net_cfg.add_layer(LayerConfig::new(
        "2",
        LayerType::Linear(LinearConfig { output_size: 1 }),
    ));
    net_cfg.add_layer(LayerConfig::new("2s", LayerType::ReLU));

    let mut err_cfg = SequentialConfig::default();
    err_cfg.add_input("nout", &[4, 1]);
    err_cfg.add_input("real", &[4, 1]);
    err_cfg.add_layer(LayerConfig::new("s2", LayerType::MeanSquaredError));

    let solv_cfg = SolverConfig {
        name: "solver".to_owned(),
        network: LayerConfig::new("net", net_cfg),
        objective: LayerConfig::new("err", err_cfg),
        ..Default::default()
    };
    let mut solver = Solver::from_config(back.clone(), back.clone(), &solv_cfg);

    let v_inp = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let v_out = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let (t_inp, t_out) = {
        let inp = Arc::new(RwLock::new(SharedTensor::<f32>::new(&[4,2])));
        let out = Arc::new(RwLock::new(SharedTensor::<f32>::new(&[4,1])));

        {
            let mut inp_p = inp.write().unwrap();
            let mut out_p = out.write().unwrap();

            for i in 0..4 {
                write_batch_sample(&mut inp_p, &v_inp[i], i);
                write_batch_sample(&mut out_p, &v_out[i], i);
            }
        }

        (inp, out)
    };

    for i in 0..1000000 {
        solver.train_minibatch(t_inp.clone(), t_out.clone());
        if i%10000 == 0 {
            let out = solver.mut_network().forward(&[t_inp.clone()]).clone();
            let out2 : Vec<_> = out.iter().map(|x| x.read().unwrap()).collect();
            let out3 : Vec<_> = out2.iter().map(|x| x.read(back.device()).unwrap().as_slice::<f32>().to_owned()).collect();
            eprintln!("{}: {:?}", i/10000, out3);
        }
    }

    let out = solver.mut_network().forward(&[t_inp.clone()]).clone();
    let out2 : Vec<_> = out.iter().map(|x| x.read().unwrap()).collect();
    let out3 : Vec<_> = out2.iter().map(|x| x.read(back.device()).unwrap().as_slice::<f32>().to_owned()).collect();
    eprintln!("{:?}", out3);
}
