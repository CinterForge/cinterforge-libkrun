pub mod epoll;
pub mod eventfd;

/// Run the calling thread at QOS_CLASS_USER_INTERACTIVE. macOS threads start at the
/// default QoS whatever their creator runs at; the vCPUs and the GPU worker are on the
/// guest's frame path and should get P-cores and priority like a game's threads.
pub fn set_thread_qos_user_interactive() {
    // SAFETY: only changes the calling thread's scheduling class.
    unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
}
