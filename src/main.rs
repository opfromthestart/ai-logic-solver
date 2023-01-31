use std::{
    fmt::{Debug, Display},
    rc::Rc,
    sync::{Arc, RwLock},
};

use coaster::{
    frameworks::{
        cuda::get_cuda_backend,
        native::{get_native_backend, Cpu},
    },
    SharedTensor, Backend, Cuda,
};
use itertools::Itertools;
use juice::{
    layer::{ LayerConfig,
        LayerType, Layer,
    },
    layers::{LinearConfig, SequentialConfig},
    solver::{Solver, SolverConfig},
};
use rand::{seq::SliceRandom, thread_rng, RngCore};

trait Board: Display + Debug + Clone {
    fn as_floats(&self) -> Vec<f32>;

    fn len(&self) -> usize;

    fn update(&mut self, new: Vec<f32>);

    fn rand(size: usize, r: &mut impl RngCore) -> Self;

    fn hide(&self, r: &mut impl RngCore, frac: f32) -> Self;

    fn as_shared(&self, dev: &Cpu) -> Arc<RwLock<SharedTensor<f32>>> {
        let data = self.as_floats();
        let mut share = SharedTensor::new(&[self.len()]);
        share
            .write_only(dev)
            .unwrap()
            .as_mut_slice::<f32>()
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = data[i]);
        Arc::new(RwLock::new(share))
    }
}

#[derive(Clone)]
struct Futoshiki {
    board: Vec<f32>,
    size: usize,
}

impl Futoshiki {
    fn new(size: usize) -> Self {
        Self {
            board: vec![0.0; size * size * size],
            size,
        }
    }
}

impl Display for Futoshiki {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.size;
        for x in 0..s {
            for y in 0..s {
                let mut m = 1.0 - 2.0 / (s as f32);
                let mut mp = None;
                for d in 0..s {
                    if self.board[x * s * s + y * s + d] > m {
                        m = self.board[x * s * s + y * s + d];
                        mp = Some(d);
                    }
                }
                match mp {
                    Some(n) => write!(f, "{} ", n)?,
                    None => write!(f, "? ")?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Debug for Futoshiki {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Futoshiki").field("size", &self.size).finish()?;
        let s = self.size;
        for x in 0..s {
            for y in 0..s {
                write!(f, "[")?;
                for z in 0..s {
                    write!(f, "{}, ", self.board[x*s*s+y*s+z])?;
                }
                write!(f, "]")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

fn perms(s: usize) -> Vec<Vec<usize>> {
    (0..s).permutations(s).collect_vec()
}

impl Board for Futoshiki {
    fn as_floats(&self) -> Vec<f32> {
        self.board.clone()
    }

    fn len(&self) -> usize {
        self.size.pow(3)
    }

    fn update(&mut self, new: Vec<f32>) {
        if new.len() == self.len() {
            self.board = new;
        } else {
            panic!("Wrong size");
        }
    }

    fn rand(s: usize, r: &mut impl RngCore) -> Self {
        let mut tries = 0;
        let perms = perms(s);
        'big: loop {
            tries += 1;
            if tries % 1000000 == 0 {
                println!("{tries}");
            }
            let rows: Vec<_> = perms.choose_multiple(r, s).collect();
            for x in 0..s {
                for y in 0..s {
                    for d in 0..s {
                        if d != x && rows[x][y] == rows[d][y] {
                            continue 'big;
                        }
                        if d != y && rows[x][y] == rows[x][d] {
                            continue 'big;
                        }
                    }
                }
            }
            let data = rows
                .iter()
                .cloned();
            //eprintln!("{data:?}");
            let data = data.flatten()
                .map(|x| (0..s).map(move |y| if x == &y { 1.0 } else { -1.0 }))
                .flatten()
                .collect_vec();
            return Self {
                board: data,
                size: s,
            };
        }
    }

    fn hide(&self, r: &mut impl RngCore, frac: f32) -> Self {
        let board = self
            .board
            .iter()
            .chunks(self.size).into_iter()
            .map(|x| {
                if (r.next_u32() as f32) / 4294967296.0 > frac {
                    x.into_iter().cloned().collect()
                } else {
                    vec![0.0; self.size]
                }
            })
            .flatten()
            .collect();
        Self {
            board,
            size: self.size,
        }
    }
}

fn main() {
    let back_cuda = Rc::new(get_cuda_backend());
    let back_nat = Rc::new(get_native_backend());

    let size = 4;
    let mut frac = 0.03;
    //let decay = 0.99;
    let mut r = thread_rng();

    println!("{}", Futoshiki::rand(size, &mut r));

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[1, size * size * size]);
    net_cfg.add_layer(LayerConfig::new(
        "1",
        LayerType::Linear(LinearConfig {
            output_size: 2 * size * size * size,
        }),
    ));
    net_cfg.add_layer(LayerConfig::new("1s", LayerType::ReLU));
    net_cfg.add_layer(LayerConfig::new(
        "2",
        LayerType::Linear(LinearConfig {
            output_size: 2 * size * size * size,
        }),
    ));
    net_cfg.add_layer(LayerConfig::new("2s", LayerType::ReLU));
    net_cfg.add_layer(LayerConfig::new(
        "3",
        LayerType::Linear(LinearConfig {
            output_size: size * size * size,
        }),
    ));
    net_cfg.add_layer(LayerConfig::new("3s", LayerType::TanH));

    let mut err_cfg = SequentialConfig::default();
    err_cfg.add_input("nout", &[size * size * size]);
    err_cfg.add_input("real", &[size * size * size]);
    err_cfg.add_layer(LayerConfig::new("s2", LayerType::MeanSquaredError));

    let solv_cfg = SolverConfig {
        name: "solver".to_owned(),
        network: LayerConfig::new("net", net_cfg),
        objective: LayerConfig::new("err", err_cfg),
        ..Default::default()
    };
    let mut solver = Solver::from_config(back_cuda.clone(), back_cuda.clone(), &solv_cfg);
    if let Ok(layer) = Layer::<Backend<Cuda>>::load(back_cuda.clone(), "saves/futo.net") {
        solver.worker.init(&layer);
        *(solver.mut_network()) = layer;
        println!("Loaded");
    }
    else {
        println!("Did not load");
    }

    for epoch in 0..10_000_000 {
        let board = Futoshiki::rand(size, &mut r);
        let out = board.as_shared(back_nat.device());
        let inp = board.hide(&mut r, frac).as_shared(back_nat.device());

        solver.train_minibatch(inp, out);
        if epoch % 20_000 == 0 {
            let mut err = 0.0;
            for _i in 0..100 {
                let board = Futoshiki::rand(size, &mut r);
                let out = board.as_floats();
                let inp = board.hide(&mut r, frac).as_shared(back_nat.device());

                let out1 = solver.mut_network().forward(&[inp.clone()])[0].clone();
                let out2 = out1.read().unwrap();
                let out3 = out2
                    .read(back_nat.device())
                    .unwrap()
                    .as_slice::<f32>()
                    .to_owned();

                //eprintln!("{},{}", out3.len(), out.len());
                let errs = out3
                .iter()
                .zip(out.iter())
                .map(|(t, a)| (t - a) * (t - a));
                //if _i == 0 {
                //    eprintln!("{:?}", errs.clone().collect::<Vec<_>>());
                //}
                err += errs.sum::<f32>();
                //eprintln!("{err}");
                //eprintln!("{}: {:?}", epoch/10000, out3);
            }
            err /= 100.;
            frac = (-err/20.).exp();
            eprintln!("{epoch}: {err}, {frac}");

            let mut board = Futoshiki::rand(size, &mut r);
            //eprintln!("{board}");
            let hidden = board.hide(&mut r, frac);
            //eprintln!("{hidden}");
            //eprintln!("{hidden:?}");
            let hidden2 = hidden.clone();
            let inp = hidden2.as_shared(back_nat.device());

            let out1 = solver.mut_network().forward(&[inp.clone()])[0].clone();
            let out2 = out1.read().unwrap();
            let out3 = out2
                .read(back_nat.device())
                .unwrap()
                .as_slice::<f32>()
                .to_owned();
            board.update(out3);
            //eprintln!("{board}");
            //eprintln!("{board:?}");
        }
        if (epoch+1) % 20_000 == 0 {
            solver.mut_network().save("saves/futo.net").expect("Could not write to file");
            //println!("Saved");
        }
    }

    let board = Futoshiki::rand(size, &mut r);
    let inp = board.hide(&mut r, frac).as_shared(back_nat.device());

    let out = solver.mut_network().forward(&[inp.clone()]).clone();
    let out2: Vec<_> = out.iter().map(|x| x.read().unwrap()).collect();
    let out3: Vec<_> = out2
        .iter()
        .map(|x| {
            x.read(back_nat.device())
                .unwrap()
                .as_slice::<f32>()
                .to_owned()
        })
        .collect();
    eprintln!("{out3:?}");
}
