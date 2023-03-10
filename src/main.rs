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
//use juice::capnp_util::{CapnpWrite, CapnpRead};
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

fn as_tensor<T: Clone>(dev: &Cpu, data: &[T]) -> Arc<RwLock<SharedTensor<T>>> {
    let mut share = SharedTensor::new(&[data.len()]);
    share
        .write_only(dev)
        .unwrap()
        .as_mut_slice::<T>()
        .iter_mut()
        .enumerate()
        .for_each(|(i, p)| *p = data[i].clone());
    Arc::new(RwLock::new(share))
}

fn from_tensor<T: Clone>(dev: &Cpu, data: &Arc<RwLock<SharedTensor<T>>>) -> Vec<T> {
    data.read().unwrap().read(dev).unwrap().as_slice::<T>().to_owned()
}

#[derive(Clone)]
enum Comp {
    None,
    Small,
    Large,
}

impl Comp {
    fn val(&self) -> f32 {
        match self {
            Comp::None => 0.0,
            Comp::Small => -0.5,
            Comp::Large => 0.5,
        }
    }

    fn charh(&self) -> char {
        match self {
            Comp::None => ' ',
            Comp::Small => '<',
            Comp::Large => '>',
        }
    }

    fn charv(&self) -> char {
        match self {
            Comp::None => ' ',
            Comp::Small => '^',
            Comp::Large => 'v',
        }
    }
}

#[derive(Clone)]
struct Futoshiki {
    board: Vec<f32>,
    comp: Vec<Comp>,
    size: usize,
}

impl Futoshiki {
    fn new(size: usize) -> Self {
        Self {
            board: vec![0.0; size * size * size],
            comp: vec![Comp::None; 2*size*(size-1)],
            size,
        }
    }
}

impl Display for Futoshiki {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.size;
        for x in 0..s {
            for y in 0..s {
                let mut m = 0.5 - 1.0 / (s as f32);
                let mut mp = None;
                for d in 0..s {
                    if self.board[x * s * s + y * s + d] > m {
                        m = self.board[x * s * s + y * s + d];
                        mp = Some(d);
                    }
                }
                let c = if y==s-1 { ' ' }
                else {
                    self.comp[x*(s-1)+y].charh()
                };
                match mp {
                    Some(n) => write!(f, "{}{}", n, c)?,
                    None => write!(f, "?{}", c)?,
                }
            }
            writeln!(f)?;
            if x != s-1 {
                for y in 0..s-1 {
                    write!(f, "{} ", self.comp[s*(s-1)+y*(s-1)+x].charv())?;
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
        let mut b = self.board.clone();
        b.extend(self.comp.iter().map(Comp::val));
        b
    }

    fn len(&self) -> usize {
        self.size.pow(3)+2*self.size*(self.size-1)
    }

    fn update(&mut self, new: Vec<f32>) {
        if new.len() == self.len() {
            self.board = new;
        } else {
            panic!("Wrong size: {}", new.len());
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
            let mut comp = vec![Comp::None; 2*s*s-1];
            for x in 0..s {
                for y in 0..s-1 {
                    if r.next_u32()%4==0 {
                        comp[x*(s-1)+y] = if rows[x][y]<rows[x][y+1] {
                            //dbg!((x,y,rows[x][y],rows[x][y+1]));
                            Comp::Small
                        }
                        else {
                            //dbg!((x,y,rows[x][y],rows[x][y+1]));
                            Comp::Large
                        };
                    }
                }
            }
            for x in 0..s-1 {
                for y in 0..s {
                    if r.next_u32()%4==0 {
                        comp[s*(s-1)+y*(s-1) + x] = if rows[x][y]<rows[x+1][y] {
                            Comp::Small
                        }
                        else {
                            Comp::Large
                        };
                    }
                }
            }
            //eprintln!("{data:?}");
            let data = data.flatten()
                .map(|x| (0..s).map(move |y| if x == &y { 0.5 } else { -0.5 }))
                .flatten()
                .collect_vec();
            return Self {
                board: data,
                comp,
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
            comp: self.comp.clone(),
            size: self.size,
        }
    }
}

fn futo_train() {
    let back_cuda = Rc::new(get_cuda_backend());
    let back_nat = Rc::new(get_native_backend());

    let size = 4;
    let mut frac = 0.03;
    //let decay = 0.99;
    let mut r = thread_rng();

    println!("{}", Futoshiki::rand(size, &mut r));

    let board_size = Futoshiki::new(size).len();

    let net_cfg = {
        let mut net_cfg=SequentialConfig::default();
        net_cfg.add_input("data", &[1, board_size]);
        //net_cfg.add_input("data2", &[1, size * size * size]);
        net_cfg.add_layer(LayerConfig::new(
        "1",
        LayerType::Linear(LinearConfig {
            output_size: 2 * board_size,
        }),
        ));
        net_cfg.add_layer(LayerConfig::new("1s", LayerType::ReLU));
        net_cfg.add_layer(LayerConfig::new(
        "2",
        LayerType::Linear(LinearConfig {
            output_size: 2 * board_size,
        }),
        ));
        net_cfg.add_layer(LayerConfig::new("2s", LayerType::ReLU));
        net_cfg.add_layer(LayerConfig::new(
        "3",
        LayerType::Linear(LinearConfig {
            output_size: board_size,
        }),
        ));
        net_cfg.add_layer(LayerConfig::new("3s", LayerType::TanH));
        net_cfg
    };

    let mut err_cfg = SequentialConfig::default();
    err_cfg.add_input("nout", &[1, board_size]);
    err_cfg.add_input("real", &[1, board_size]);
    err_cfg.add_layer(LayerConfig::new("s2", LayerType::MeanSquaredError));

    let solv_cfg = SolverConfig {
        name: "solver".to_owned(),
        network: LayerConfig::new("net", net_cfg.clone()),
        objective: LayerConfig::new("err", err_cfg),
        ..Default::default()
    };
    let mut solver = Solver::from_config(back_cuda.clone(), back_cuda.clone(), &solv_cfg);
    if let Ok(layer) = Layer::<Backend<Cuda>>::load(back_cuda.clone(), "saves/futo.net") {
        solver.worker.init(&layer);
        *(solver.mut_network()) = layer;
        println!("Loaded net");
    }
    else {
        println!("Did not load net");
    }

    for epoch in 0..10_000_000 {
        let board = Futoshiki::rand(size, &mut r);
        let out = board.as_shared(back_nat.device());
        let inp = board.hide(&mut r, frac).as_shared(back_nat.device());

        if epoch % 5_000 == 0 {
            let err_s = "Could not write to saves folder\nMake sure there is a folder in this directory named 'saves'";
            solver.mut_network().save("saves/futo.net").expect(err_s);
            //println!("Saved");
            //solver.worker.
            //let mut f = File::options().truncate(true).create(true).write(true).open("saves/futo.cfg").expect(err_s);
            // let mut builder = juice::juice_capnp::sequential_config::Builder;
            
            //let mut builder = capnp::message::TypedBuilder::<juice::juice_capnp::sequential_config::Owned>::new_default();
            //let facade = &mut builder.get_root().unwrap();
            //net_cfg.write_capnp(facade);
    
            //capnp::serialize::write_message(&mut f, builder.borrow_inner()).unwrap();
        }
        if epoch % 5_000 == 0 {
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
            let intertia = 0.9;
            frac = intertia*frac + (1.0-intertia)*(-err/8.).exp();
            eprintln!("{epoch}: {err}, {frac}");
        }
        if epoch % 50_000 == 0 {
            let mut board = Futoshiki::rand(size, &mut r);
            eprintln!("{board}");
            let hidden = board.hide(&mut r, frac);
            eprintln!("{hidden}");
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
            eprintln!("{board}");
            eprintln!("{board:?}");
        }
        else {
            solver.train_minibatch(inp, out);
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

fn main() {
    futo_train();
}
