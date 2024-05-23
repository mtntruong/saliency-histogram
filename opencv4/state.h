struct StateData;
struct State_;
typedef State_ (*State)(StateData&);
struct State_ {
	State_( State pp ) : p( pp ) { }
	operator State() { return p; }
	State p;
};

State_ state_start(StateData& d);
State_ state_selecting(StateData& d);
State_ state_initializing(StateData& d);
State_ state_tracking(StateData& d);
