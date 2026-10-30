fn move_owner_to_another_thread(owner: crate::cuda::OwnerBorrow<'static>) {
    let _ = std::thread::spawn(move || owner);
}
