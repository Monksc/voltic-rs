macro_rules! parameters {
    ($self:ident, [$( $param:ident ),*]) => {{
        let mut params = vec![];
        $( if let Some(p) = &$self.$param { params.push(p); } )*
        params
    }};
}
