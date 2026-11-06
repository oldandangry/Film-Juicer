use crate::cuda::OwnerBorrow;

fn move_callback_borrow_to_thread(owner: OwnerBorrow<'_>) {
    let _ = std::thread::spawn(move || owner);
}
